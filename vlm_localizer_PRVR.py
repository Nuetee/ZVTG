import os
import torch
import numpy as np
from scipy.optimize import minimize_scalar
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from sklearn.cluster import KMeans

#### BLIP-2 Q-Former ####
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])
#### BLIP-2 Q-Former ####


def iou(candidates, gt):
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x = np.arange(-size, size + 1)
    normal = 1 / (np.sqrt(2.0 * np.pi) * sigma)
    g = np.exp(-x ** 2 / (2.0 * sigma ** 2)) * normal
    return g

def nchk(f, f1, f2, ths):
    return (((3 * f) > ths) | ((2 * f + f1) > ths) | ((f + f1 + f2) > ths))

def get_dynamic_scores(scores, stride, masks, ths=0.0005, sigma=1):
    gstride = min(stride - 2, 3)
    if (stride < 3):
        gkernel = torch.ones((1, 1, 1)).to('cuda')
    else:
        gkernel = gaussian_kernel(gstride, sigma)
        gkernel = torch.from_numpy(gkernel).float().to('cuda')
        gkernel = gkernel.view(1, 1, -1)
    gscore = F.conv1d(scores.view(-1, 1, scores.size(-1)), gkernel).view(scores.size(0), -1)

    diffres = torch.diff(gscore).to('cuda')
    pad_left = torch.zeros((diffres.size(0), (masks.size(-1) - diffres.size(-1)) // 2)).to('cuda')
    pad_right = torch.zeros((diffres.size(0), masks.size(-1) - diffres.size(-1) - pad_left.size(-1))).to('cuda')
    diffres = torch.cat((pad_left, diffres, pad_right), dim=-1) * masks

    dynamic_scores = np.zeros((diffres.size(0), diffres.size(-1)))
    dynamic_idxs = np.zeros((diffres.size(0), diffres.size(-1)))

    for idx in range(diffres.size(0)):
        f1 = f2 = f3 = 0
        d_score = 0
        d_idx = 0
        for i in range(diffres.size(-1)):
            f3 = f2
            f2 = f1
            f1 = diffres[idx][i]
            if nchk(f1, f2, f3, ths):
                d_score += max(3 * f1, 2 * f1 + f2, f1 + f2 + f3)
            else:
                d_idx = i
                d_score = 0

            dynamic_idxs[idx][i] = d_idx / scores.size(-1)
            dynamic_scores[idx][i] = d_score

    dynamic_idxs = torch.from_numpy(dynamic_idxs).to('cuda')
    dynamic_scores = torch.from_numpy(dynamic_scores).to('cuda')
    return dynamic_idxs, dynamic_scores


def split_interval(init_timestep):
    init_timestep = init_timestep.cpu().sort()[0]
    # 결과를 저장할 리스트
    ranges = []

    # 임시로 시작과 끝을 저장할 변수
    start = init_timestep[0]
    end = init_timestep[0].clone()

    # 텐서의 각 원소를 순차적으로 비교
    for i in range(1, len(init_timestep)):
        if init_timestep[i] == end + 1:
            # 연속된 숫자인 경우
            end = init_timestep[i]
        else:
            # 연속되지 않은 숫자가 나타나면 구간을 추가하고 새로 시작
            ranges.append([start, end])
            start = init_timestep[i]
            end = init_timestep[i].clone()

    # 마지막 구간 추가
    ranges.append([start, end])
    return torch.tensor(ranges)

import re
def sanitize_filename(filename):
    # 허용되지 않는 문자를 `_`로 대체
    filename = re.sub(r'[\/:*?"<>|]', '_', filename)
    return filename


def extract_static_score(start, end, cum_scores, num_frames):
    kernel_size = end - start
    if start == 0:
        inner_sum = cum_scores[end - 1]
    else:
        inner_sum = cum_scores[end - 1] - cum_scores[start - 1]

    outer_sum = cum_scores[num_frames - 1] - inner_sum

    if kernel_size != num_frames:
        static_score = inner_sum / kernel_size - outer_sum / (num_frames - kernel_size)
    else:
        # static_score = inner_sum / kernel_size - (scores[0][0] + scores[0][-1] / 2)
        static_score = inner_sum / kernel_size
    return static_score


def scores_masking(scores, masks):
    # scores의 길이가 3 미만인 경우 initial_mask를 그대로 사용
    if scores.shape[1] < 3:
        masks = masks.squeeze()
    else:
        # 양쪽 끝에 2씩 False로 패딩
        padded_masks = F.pad(masks, (1, 1), mode='constant', value=False)

        # 현재 위치를 기준으로 양옆 2개의 값 기반 Majority voting, 최종 마스크 결과 저장
        final_masks = padded_masks.clone()
        for i in range(2, padded_masks.shape[1] - 1):
            window = padded_masks[:, i - 1 : i + 2]
            if window.sum() < 2:
                final_masks[:, i] = 0

        # 패딩 제거하여 원래 크기의 마스크로 복원
        masks = final_masks[:, 1:-1].squeeze()
    
    # 모든 값이 False일 경우 전부 True로 설정
    if not masks.any():
        masks[:] = True

    # final_mask를 기반으로 masked_indices 계산
    masked_indices = torch.nonzero(masks, as_tuple=True)[0]  # 마스킹된 실제 인덱스 저장
    
    return masks, masked_indices


def alignment_adjustment(data, scale_gamma, device, lambda_max=2, lambda_min=-2):
    # 작은 상수 추가로 양수 데이터 보장
    epsilon = 1e-6
    data = data + abs(data.min()) + epsilon if np.any(data <= 0) else data
    
    def boxcox_transformed(x, lmbda):
        if lmbda == 0:
            return np.log(x)
        else:
            return (x**lmbda - 1) / lmbda

    # 최적의 lambda를 찾기 위한 로그 가능도 함수 (최소화할 함수)
    def neg_log_likelihood(lmbda):
        transformed_data = boxcox_transformed(data, lmbda)
        # 분산 계산 시 overflow 방지
        var = np.var(transformed_data, ddof=1)
        return -np.sum(np.log(np.abs(transformed_data))) + 0.5 * len(data) * np.log(var)

    # lambda 범위 내에서 최적화
    result = minimize_scalar(neg_log_likelihood, bounds=(lambda_min, lambda_max), method='bounded')
    best_lambda = result.x
    
    # 최적의 lambda로 변환 데이터 생성
    transformed_data = boxcox_transformed(data, best_lambda)

    original_min, original_max = data.min(), data.max()
    transformed_min, transformed_max = transformed_data.min(), transformed_data.max()
    transformed_data = (transformed_data - transformed_min) / (transformed_max - transformed_min)  # normalize to [0, 1]
    is_scale = False
    if original_max - original_min > scale_gamma:
        is_scale = True
        transformed_data = transformed_data * (original_max - original_min) + original_min  # scale to original min/max
    else:
        transformed_data = transformed_data * (scale_gamma) + original_min
    # 변환 결과를 다시 텐서로 변환하고 원래 형태로 복원

    normalized_scores = torch.tensor(transformed_data, device=device).unsqueeze(0)

    return normalized_scores, is_scale


import torch.nn.functional as F
def temporal_aware_feature_smoothing(kernel_size, video_features):
    F_dim, Q, C = video_features.shape  # Frame, Query, Channel 크기

    # padding size 계산
    pad_size = kernel_size // 2

    video_features = video_features.permute(1, 2, 0)  # (Q, C, F)

    # Frame 차원(F)에 대해 padding 적용
    padded_features = F.pad(video_features, (pad_size, pad_size), mode='replicate')  # (Q, C, F + 2*pad_size)

    # Mean pooling을 Frame 차원(F)에서 수행
    pooled_features = F.avg_pool1d(padded_features, kernel_size, stride=1)  # (Q, C, F)

    # 원래 차원 (F, Q, C)으로 변환
    return pooled_features.permute(2, 0, 1)  # (F, Q, C)


def kmeans_clustering(k, features):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(np.array(features.cpu()))
    kmeans_labels = torch.tensor(kmeans_labels)

    return kmeans_labels


def kmeans_clustering_gpu(k, features, n_iter=100, tol=1e-4):
    # Ensure features are on GPU
    torch.manual_seed(60)
    features = features.cuda().float()
    features = features.mean(dim=1)
    n_samples, n_features = features.shape

    # Initialize centroids using k-means++ algorithm
    centroids = torch.empty((k, n_features), device=features.device, dtype=torch.float32)
    # Step 1: Choose the first centroid randomly
    random_idx = torch.randint(0, n_samples, (1,))
    centroids[0] = features[random_idx]

    # Step 2: Choose remaining centroids
    for i in range(1, k):
        # Compute squared distances from the closest centroid
        distances = torch.min(torch.cdist(features, centroids[:i])**2, dim=1).values
        probabilities = distances / distances.sum()
        cumulative_probs = torch.cumsum(probabilities, dim=0)
        random_value = torch.rand(1, device=features.device)
        next_idx = torch.searchsorted(cumulative_probs, random_value).item()
        centroids[i] = features[next_idx]

    # Perform k-means clustering
    for i in range(n_iter):
        # Calculate distances (broadcasting)
        distances = torch.cdist(features, centroids, p=2)

        # Assign clusters
        labels = torch.argmin(distances, dim=1)

        # Update centroids
        new_centroids = torch.stack([features[labels == j].mean(dim=0) if (labels == j).sum() > 0 else centroids[j] for j in range(k)])

        # Check for convergence
        if torch.allclose(centroids, new_centroids, atol=tol):
            break

        centroids = new_centroids

    return labels.cpu()

def segment_scenes_by_cluster(cluster_labels):
    scene_segments = []
    start_idx = 0

    current_label = cluster_labels[0]
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != current_label:
            scene_segments.append([start_idx, i])  ### start_idx 이상, i 미만 까지 같은 레이블
            start_idx = i
            current_label = cluster_labels[i]
    
    scene_segments.append([start_idx, len(cluster_labels)])
    scene_segments.append([len(cluster_labels), len(cluster_labels)])

    return scene_segments


def generate_segments(video_features, hyperparams, kmeans_gpu):
    num_frames = video_features.shape[0]

    video_features = torch.tensor(video_features).cuda()
        
    # Temporal-aware vector smoothing
    temporal_aware_features = temporal_aware_feature_smoothing(hyperparams['temporal_window_size'], video_features)

    # Kmeans Clustering
    kmeans_k = min(hyperparams['kmeans_k'], max(2, num_frames))
    if kmeans_gpu:
        kmeans_labels = kmeans_clustering_gpu(kmeans_k, temporal_aware_features)
    else:
        kmeans_labels = kmeans_clustering(kmeans_k, temporal_aware_features)
    
    # Kmeans clusetring 결과에 따라 비디오 장면 Segmentation
    scene_segments = segment_scenes_by_cluster(kmeans_labels)
    return scene_segments, num_frames


def get_proposals_with_scores(scene_segments, cum_scores, cum_norm_scores, num_frames, prior):
    proposals = []
    proposals_inner_mean = []
    proposals_norm_scores = []
    for i in range(len(scene_segments)):
        for j in range(i + 1, len(scene_segments)):
            start = scene_segments[i][0]
            last = scene_segments[j][0]
            if (last - start) > num_frames * prior:
                continue
            score_norm = extract_static_score(start, last, cum_norm_scores, len(cum_norm_scores)).item()
            
            kernel_size = last - start
            if start == 0:
                inner_sum = cum_scores[last - 1]
            else:
                inner_sum = cum_scores[last - 1] - cum_scores[start - 1]
            
            inner_mean = inner_sum / kernel_size
            inner_mean = inner_mean.item()
            
            proposals.append([start, last])
            proposals_norm_scores.append(round(score_norm, 4))
            proposals_inner_mean.append(round(inner_mean, 4))

    return proposals, proposals_norm_scores, proposals_inner_mean


def video_retrieval(data, feature_path, query, stride, hyperparams, kmeans_gpu):
    video_scores = []
    with torch.no_grad():
        text = model.tokenizer(query, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
            'cuda')
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
    v1 = F.normalize(text_feat, dim=-1)

    for vid, ann in data.items():
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        num_frames = video_feature.shape[0]
        duration = ann['duration']

        v2 = F.normalize(torch.tensor(video_feature, device='cuda', dtype=v1.dtype), dim=-1)
        scores = torch.einsum('md,npd->mnp', v1, v2)
        scores, scores_idx = scores.max(dim=-1)
        scores = scores.mean(dim=0, keepdim=True)
        # scores > 0.2인 마스킹 생성 (Boolean 형태 유지)
        initial_masks = (scores > 0.2 if hyperparams['is_blip2'] else scores > 0)
        masks, masked_indices = scores_masking(scores, initial_masks)

        # Alignment adjustment of similarity scores
        data = scores[:, masks].flatten().cpu().numpy()   # 마스크된 부분만 가져오기    
        normalized_scores, is_scale = alignment_adjustment(data, hyperparams['gamma'], scores.device, lambda_max=2, lambda_min=-2)

        # 마스킹된 부분을 0으로 채운 원본 크기의 scores 텐서 생성
        restored_scores = torch.zeros_like(scores)  # 원본 scores 크기로 초기화

        # 마스킹된 부분만 normalized_scores 값으로 채우기
        restored_scores[:, masks] = normalized_scores.clone().detach()
        cum_norm_scores = torch.cumsum(restored_scores, dim=1)[0]
        cum_scores = torch.cumsum(scores, dim=1)[0]

        scene_segments = ann['scene_segments']
        final_proposals, final_proposals_score, final_proposals_inner_mean = get_proposals_with_scores(scene_segments, cum_scores, cum_norm_scores, num_frames, hyperparams['prior'])
        
        final_proposals = torch.tensor(final_proposals)
        final_proposals_score = torch.tensor(final_proposals_score)
        final_proposals_inner_mean = torch.tensor(final_proposals_inner_mean)
        _, score_index = final_proposals_score.sort(descending=True)
        final_proposals = final_proposals[score_index]
        final_proposals_scores = final_proposals_score[score_index]
        final_proposals_inner_mean = final_proposals_inner_mean[score_index]

        #### dynamic scoring #####
        masked_scores = scores * initial_masks.float()
        stride = min(stride, masked_scores.size(-1) // 2)

        dynamic_idxs, dynamic_scores = get_dynamic_scores(masked_scores, stride, initial_masks.float())
        dynamic_frames = torch.round(dynamic_idxs * num_frames).int()
        
        for final_proposal in final_proposals:
            current_frame = final_proposal[0]
            dynamic_prefix = dynamic_frames[0][current_frame]
            while True:
                if current_frame == 0 or dynamic_frames[0][current_frame - 1] != dynamic_prefix:
                    break
                current_frame -= 1
            final_proposal[0] = current_frame

        #### dynamic scoring #####
        
        if len(final_proposals) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = final_proposals / num_frames * duration
            scores = final_proposals_scores
            scores = scores / scores.max()

        proposals = []
        for s_pred, score, inner_mean in zip(static_pred, scores, final_proposals_inner_mean):
            proposals.append([
                float(s_pred[0]),
                float(s_pred[1]),
                float(score),
                float(inner_mean)
            ])

        proposals = proposals[:hyperparams['cand_num']]

        def calc_iou(candidates, gt):
            start, end = candidates[:,0], candidates[:,1]
            s, e = gt[0], gt[1]
            inter = np.minimum(end, e) - np.maximum(start, s)
            union = np.maximum(end, e) - np.minimum(start, s)
            return inter.clip(min=0) / union
        
        def select_proposal(inputs, gamma=0.6):
            weights = inputs[:, 2].clip(min=0)
            proposals = inputs[:, :2]
            scores = np.zeros_like(weights)

            for j in range(scores.shape[0]):
                iou = calc_iou(proposals, proposals[j])
                scores[j] += (iou ** gamma * weights).sum()

            idx = np.argsort(-scores)
            return inputs[idx]

        proposals = select_proposal(np.array(proposals))
        video_scores.append({
            'vid': vid,
            'score': proposals[0][-1]
        })
    
    sorted_vids = [item['vid'] for item in sorted(video_scores, key=lambda x: x['score'], reverse=True)]

    return sorted_vids


def get_proposals_with_scores_batch(scene_segments, cum_scores, num_frames, prior):
    proposals = []
    proposals_static_scores = []
    proposals_inner_mean = []
    for i in range(len(scene_segments)):
        for j in range(i + 1, len(scene_segments)):
            start = scene_segments[i][0]
            last = scene_segments[j][0]
            if (last - start) > num_frames * prior:
                continue
            score_static = extract_static_score(start, last, cum_scores, len(cum_scores)).item()

            kernel_size = last - start
            if start == 0:
                inner_sum = cum_scores[last - 1]
            else:
                inner_sum = cum_scores[last - 1] - cum_scores[start - 1]
            
            inner_mean = inner_sum / kernel_size
            inner_mean = inner_mean.item()
            
            proposals.append([start, last])
            proposals_static_scores.append(round(score_static, 4))
            proposals_inner_mean.append(round(inner_mean, 4))

    return proposals, proposals_static_scores, proposals_inner_mean


def video_retrieval_batch(data, feature_path, query, stride, hyperparams, batch_size=32):
    all_results = []

    # 1️⃣ Query를 BLIP-2 모델로 처리 (한 번만 수행)
    with torch.no_grad():
        text = model.tokenizer(query, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])  # (1, D)
        text_feat = F.normalize(text_feat, dim=-1)  # (1, D)
    
    # 2️⃣ 데이터셋을 batch_size 단위로 처리
    data_items = list(data.items())
    num_batches = (len(data_items) + batch_size - 1) // batch_size  # 올림 처리
    
    for batch_idx in range(num_batches):
        batch_data = data_items[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        video_features = []
        video_meta = []  # (vid, duration, scene_segments)
        frame_counts = []

        # 3️⃣ 현재 batch의 비디오 feature 로드
        for vid, ann in batch_data:
            feature = np.load(os.path.join(feature_path, vid + '.npy'))  # (F, Q, D)
            video_features.append(torch.tensor(feature, dtype=torch.float32, device='cuda'))
            frame_counts.append(feature.shape[0])  # 각 비디오의 실제 Frame 수 저장
            video_meta.append((vid, ann['duration'], ann['scene_segments']))
        
        # 4️⃣ Padding: 가장 긴 비디오의 Frame 수에 맞춤
        max_F = max(frame_counts)  # 배치 내 가장 긴 비디오의 Frame 수
        padded_video_features = []
        padding_masks = []

        for i, feature in enumerate(video_features):
            F_c, Q, D = feature.shape
            pad_size = max_F - F_c

            # 뒤쪽에 패딩 추가 (zero-padding)
            padded_feature = F.pad(feature, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
            padded_video_features.append(padded_feature)

            # 실제 프레임 여부를 나타내는 mask 생성
            mask = torch.cat([torch.ones(F_c, device='cuda'), torch.zeros(pad_size, device='cuda')])
            padding_masks.append(mask)
        
        # (B, max_F, Q, D) 형태로 변환
        video_features = torch.stack(padded_video_features)
        padding_masks = torch.stack(padding_masks)  # (B, max_F)

        B, max_F, Q, D = video_features.shape  # 이제 모든 비디오의 Frame 수가 동일

        # 5️⃣ Query와 batch 내 모든 비디오 간 코사인 유사도 계산
        video_features = F.normalize(video_features, dim=-1)  # (B, max_F, Q, D)
        scores = torch.einsum('md,bfqd->bfq', text_feat, video_features)  # (B, max_F, Q)

        # 6️⃣ Q 차원에서 최댓값을 취해서 frame-level 유사도 만들기
        scores, _ = scores.max(dim=-1)  # (B, max_F)

        # 7️⃣ 패딩된 부분을 제외하기 위해 mask 적용
        scores = scores * padding_masks  # (B, max_F), 패딩된 부분을 0으로

        # 8️⃣ 마스크 생성 및 정규화 (수정된 버전)
        masks = scores > (0.2 if hyperparams['is_blip2'] else 0)
        scores *= masks
        
        # 🔟 proposals 생성
        batch_results = []
        for i in range(B):
            vid, duration, scene_segments = video_meta[i]
            num_frames = frame_counts[i]  # 원래 비디오의 Frame 수 사용

            cum_scores = torch.cumsum(scores[i][:num_frames], dim=0)

            final_proposals, final_proposals_score, final_proposals_inner_mean = get_proposals_with_scores_batch(
                scene_segments, cum_scores, num_frames, hyperparams['prior']
            )

            final_proposals = torch.tensor(final_proposals)
            final_proposals_score = torch.tensor(final_proposals_score)
            final_proposals_inner_mean = torch.tensor(final_proposals_inner_mean)
            _, score_index = final_proposals_score.sort(descending=True)
            final_proposals = final_proposals[score_index]
            final_proposals_scores = final_proposals_score[score_index]
            final_proposals_inner_mean = final_proposals_inner_mean[score_index]

            # 정규화된 proposal 생성
            if len(final_proposals) == 0:
                static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
                proposal_scores  = np.array([1.0, 1.0, 1.0])
            else:
                static_pred = final_proposals / num_frames * duration
                proposal_scores  = final_proposals_scores
                proposal_scores  = proposal_scores  / proposal_scores .max()

            proposals = []
            for s_pred, p_score, inner_mean in zip(static_pred, proposal_scores , final_proposals_inner_mean):
                proposals.append([
                    float(s_pred[0]),
                    float(s_pred[1]),
                    float(p_score),
                    float(inner_mean)
                ])

            proposals = proposals[:hyperparams['cand_num']]

            def calc_iou(candidates, gt):
                start, end = candidates[:, 0], candidates[:, 1]
                s, e = gt[0], gt[1]
                inter = np.minimum(end, e) - np.maximum(start, s)
                union = np.maximum(end, e) - np.minimum(start, s)
                return inter.clip(min=0) / union
            
            def select_proposal(inputs, gamma=0.6):
                weights = inputs[:, 2].clip(min=0)
                proposals = inputs[:, :2]
                scores = np.zeros_like(weights)

                for j in range(scores.shape[0]):
                    iou = calc_iou(proposals, proposals[j])
                    scores[j] += (iou ** gamma * weights).sum()

                idx = np.argsort(-scores)
                return inputs[idx]

            proposals = select_proposal(np.array(proposals))
            batch_results.append({
                'vid': vid,
                'score': proposals[0][-1]
            })

        # 11️⃣ 현재 batch의 결과 저장
        all_results.extend(batch_results)
    
    # 12️⃣ 최종 정렬 및 반환
    sorted_vids = [item['vid'] for item in sorted(all_results, key=lambda x: x['score'], reverse=True)]
    return sorted_vids


import itertools

def video_retrieval_batch2(data, feature_path, query, stride, hyperparams, batch_size=32):
    all_results = []

    # 1️⃣ Query를 BLIP-2 모델로 처리 (한 번만 수행)
    with torch.no_grad():
        text = model.tokenizer(query, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])  # (1, D)
        text_feat = F.normalize(text_feat, dim=-1)  # (1, D)

    # 2️⃣ 데이터셋을 batch_size 단위로 처리
    data_items = list(data.items())
    num_batches = (len(data_items) + batch_size - 1) // batch_size  # 올림 처리

    for batch_idx in range(num_batches):
        batch_data = data_items[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        video_features = []
        video_meta = []  # (vid, duration, segment_boundaries)
        frame_counts = []

        # 3️⃣ 현재 batch의 비디오 feature 로드
        for vid, ann in batch_data:
            feature = np.load(os.path.join(feature_path, vid + '.npy'))  # (F, Q, D)
            video_features.append(torch.tensor(feature, dtype=torch.float32, device='cuda'))
            frame_counts.append(feature.shape[0])  # 각 비디오의 실제 Frame 수 저장
            video_meta.append((vid, ann['duration'], ann['segment_boundaries']))

        # 4️⃣ Padding: 가장 긴 비디오의 Frame 수에 맞춤
        max_F = max(frame_counts)  # 배치 내 가장 긴 비디오의 Frame 수
        padded_video_features = []
        padding_masks = []

        for i, feature in enumerate(video_features):
            F_c, Q, D = feature.shape
            pad_size = max_F - F_c

            # 뒤쪽에 패딩 추가 (zero-padding)
            padded_feature = F.pad(feature, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
            padded_video_features.append(padded_feature)

            # 실제 프레임 여부를 나타내는 mask 생성
            mask = torch.cat([torch.ones(F_c, device='cuda'), torch.zeros(pad_size, device='cuda')])
            padding_masks.append(mask)

        # (B, max_F, Q, D) 형태로 변환
        video_features = torch.stack(padded_video_features)
        padding_masks = torch.stack(padding_masks)  # (B, max_F)

        B, max_F, Q, D = video_features.shape  # 이제 모든 비디오의 Frame 수가 동일

        # 5️⃣ Query와 batch 내 모든 비디오 간 코사인 유사도 계산
        video_features = F.normalize(video_features, dim=-1)  # (B, max_F, Q, D)
        scores = torch.einsum('md,bfqd->bfq', text_feat, video_features)  # (B, max_F, Q)

        # 6️⃣ Q 차원에서 최댓값을 취해서 frame-level 유사도 만들기
        scores, _ = scores.max(dim=-1)  # (B, max_F)

        # 7️⃣ 패딩된 부분을 제외하기 위해 mask 적용
        scores = scores * padding_masks  # (B, max_F), 패딩된 부분을 0으로

        # 8️⃣ 마스크 생성 및 정규화
        masks = scores > (0.2 if hyperparams['is_blip2'] else 0)
        scores *= masks.float()  # (B, max_F)

        # 🔥 9️⃣ Batch-wise intra score mean 계산 (segment_boundaries 기반)
        # ① segment_boundaries의 모든 boundary를 미리 추출
        segment_boundaries_list = []
        max_boundaries = 0

        for i in range(B):
            segment_boundaries = video_meta[i][2]  # segment_boundaries 리스트
            segment_boundaries_list.append(segment_boundaries)
            max_boundaries = max(max_boundaries, len(segment_boundaries))

        # ② 모든 비디오의 `segment_boundaries`를 동일한 크기로 패딩 (Batch-wise 연산 위해)
        segment_boundaries_tensor = torch.zeros((B, max_boundaries), dtype=torch.long, device='cuda')
        for i in range(B):
            segment_boundaries_tensor[i, :len(segment_boundaries_list[i])] = torch.tensor(segment_boundaries_list[i], dtype=torch.long, device='cuda')

        # ③ 가능한 모든 (start, end) 조합을 미리 생성
        combs = list(itertools.combinations(range(max_boundaries), 2))
        start_idx_tensor = segment_boundaries_tensor[:, [c[0] for c in combs]]
        end_idx_tensor = segment_boundaries_tensor[:, [c[1] for c in combs]]  # (B, num_combinations)

        # ④ Batch-wise intra mean score 계산 (GPU 연산)
        intra_scores = torch.zeros((B, len(combs)), dtype=torch.float32, device='cuda')

        for j in range(len(combs)):  # 모든 조합에 대해 한 번씩만 연산
            start = start_idx_tensor[:, j]
            end = end_idx_tensor[:, j]
            valid_mask = start < end  # 유효한 start-end 조합만 연산
            intra_scores[:, j] = torch.where(valid_mask, torch.stack([scores[i, start[i]:end[i]].mean() for i in range(B)]), torch.tensor(0.0, device='cuda'))

        # 🔥 10️⃣ 각 비디오에서 가장 높은 intra score mean만 저장
        max_intra_scores, max_indices = intra_scores.max(dim=1)  # (B,)
        best_start_indices = start_idx_tensor[torch.arange(B, device='cuda'), max_indices]
        best_end_indices = end_idx_tensor[torch.arange(B, device='cuda'), max_indices]
        
        # 🚀 최종 결과 저장
        for i in range(B):
            all_results.append({
                'vid': video_meta[i][0],  # 비디오 ID
                'best_segment': (best_start_indices[i].item(), best_end_indices[i].item()),  # 최고 intra score mean을 가진 segment
                'best_intra_score_mean': max_intra_scores[i].item()  # 최고 intra score mean 값
            })

    sorted_vids = [item['vid'] for item in sorted(all_results, key=lambda x: x['best_intra_score_mean'], reverse=True)]
    return sorted_vids