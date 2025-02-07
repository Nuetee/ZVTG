import os
import clip
import torch
import numpy as np
from scipy.optimize import minimize_scalar
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from sklearn.cluster import KMeans


#### CLIP ####
clip_model, preprocess = clip.load("ViT-L/14", device='cuda')
clip_text_encoder = clip_model.encode_text
#### CLIP ####


#### BLIP-2 Q-Former ####
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])
#### BLIP-2 Q-Former ####


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


def extract_static_score(start, end, cum_scores, num_frames, scores):
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


def temporal_aware_feature_smoothing(kernel_size, features):
    padding_size = kernel_size // 2
    padded_features = torch.cat((features[0].repeat(padding_size, 1), features, features[-1].repeat(padding_size, 1)), dim=0)
    kernel = torch.ones(padded_features.shape[1], 1, kernel_size).cuda() / kernel_size
    padded_features = padded_features.unsqueeze(0).permute(0, 2, 1)  # (1, 257, 104)
    padded_features = padded_features.float()

    temporal_aware_features = F.conv1d(padded_features, kernel, padding=0, groups=padded_features.shape[1])
    temporal_aware_features = temporal_aware_features.permute(0, 2, 1)
    temporal_aware_features = temporal_aware_features[0]

    return temporal_aware_features


def kmeans_clustering(k, features):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_labels = kmeans.fit_predict(np.array(features.cpu()))
    kmeans_labels = torch.tensor(kmeans_labels)

    return kmeans_labels


def kmeans_clustering_gpu(k, features, n_iter=100, tol=1e-4):
    # Ensure features are on GPU
    torch.manual_seed(60)
    features = features.cuda()
    n_samples, n_features = features.shape

    # Initialize centroids using k-means++ algorithm
    centroids = torch.empty((k, n_features), device=features.device)
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


def get_proposals_with_scores(scene_segments, cum_scores, frame_scores, num_frames, prior):
    proposals = []
    proposals_static_scores = []
    for i in range(len(scene_segments)):
        for j in range(i + 1, len(scene_segments)):
            start = scene_segments[i][0]
            last = scene_segments[j][0]
            if (last - start) > num_frames * prior:
                continue
            score_static = extract_static_score(start, last, cum_scores, len(cum_scores), frame_scores).item()
            
            proposals.append([start, last])
            proposals_static_scores.append(round(score_static, 4))

    return proposals, proposals_static_scores


def generate_proposal_revise(video_features, sentences, stride, hyperparams, kmeans_gpu):
    num_frames = video_features.shape[0]
    if hyperparams['is_clip']:
        with torch.no_grad():
           text_tokens = clip.tokenize(sentences).to(device='cuda')
           text_feat = clip_text_encoder(text_tokens)
        v1 = F.normalize(text_feat, p=2, dim=1)  # Normalize along feature dimension
        v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), p=2, dim=1)  # Normalize along feature dimension
        scores = torch.matmul(v2, v1.T).squeeze()
        scores = scores.unsqueeze(0)
        video_features = torch.tensor(video_features).cuda()
    else:
        # with torch.no_grad():
        #     text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
        #         'cuda')
        #     text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        #     text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
        # v1 = F.normalize(text_feat, dim=-1)
        # v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
        # scores = torch.einsum('md,npd->mnp', v1, v2)
        # scores, scores_idx = scores.max(dim=-1)
        # scores = scores.mean(dim=0, keepdim=True)
        with torch.no_grad():
            # Tokenize text
            text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
            text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])  # (batch_size, feature_dim)

        v1 = F.normalize(text_feat, dim=-1)  # (batch_size, feature_dim)

        v2 = torch.tensor(video_features, device='cuda', dtype=v1.dtype)  # (num_frames, num_queries, feature_dim)
        v2 = v2.mean(dim=1)  # Average over num_queries -> (num_frames, feature_dim)
        v2 = F.normalize(v2, dim=-1)  # Normalize

        # Compute cosine similarity scores
        scores = torch.einsum('md,nd->mn', v1, v2)  # (batch_size, num_frames)

        # Average scores across frames
        scores = scores.mean(dim=0, keepdim=True)  

    # scores > 0.2인 마스킹 생성 (Boolean 형태 유지)
    initial_masks = (scores > 0.2 if hyperparams['is_blip2'] else scores > 0)
    masks, masked_indices = scores_masking(scores, initial_masks)
    normalized_scores = scores[:, masks]
    
    if hyperparams['is_blip2'] or hyperparams['is_blip']:
        # video_features = torch.tensor(video_features).cuda()
        # scores_idx = scores_idx.reshape(-1)
        # selected_video_features = video_features[torch.arange(num_frames), scores_idx]
        video_features = torch.tensor(video_features).cuda()
        selected_video_features = video_features.mean(dim=1) 
    else:
        selected_video_features = video_features
        
    time_features = (torch.arange(num_frames) / num_frames).unsqueeze(1).cuda()
    selected_video_time_features = torch.cat((selected_video_features, time_features), dim=1)
    selected_video_time_features = selected_video_time_features[masks]

    # Temporal-aware vector smoothing
    temporal_aware_features = temporal_aware_feature_smoothing(hyperparams['temporal_window_size'], selected_video_time_features)

    # Kmeans Clustering
    kmeans_k = min(hyperparams['kmeans_k'], max(2, len(masked_indices)))
    if kmeans_gpu:
        kmeans_labels = kmeans_clustering_gpu(kmeans_k, temporal_aware_features)
    else:
        kmeans_labels = kmeans_clustering(kmeans_k, temporal_aware_features)
    
    # Kmeans clusetring 결과에 따라 비디오 장면 Segmentation
    scene_segments = segment_scenes_by_cluster(kmeans_labels)

    # proposal generation by using scene segments integration
    cum_scores = torch.cumsum(normalized_scores, dim=1)[0]
    final_proposals, final_proposals_static_score = get_proposals_with_scores(scene_segments, cum_scores, normalized_scores, num_frames, hyperparams['prior'])

    final_proposals = [
        [
            masked_indices[start].item() if start < len(masked_indices) else num_frames,
            masked_indices[last].item() if last < len(masked_indices) else num_frames
        ]
        for start, last in final_proposals
    ]
    final_proposals = torch.tensor(final_proposals)
    final_proposals_static_score = torch.tensor(final_proposals_static_score)
    _, index_static = final_proposals_static_score.sort(descending=True)
    final_proposals = final_proposals[index_static]
    final_proposals_scores = final_proposals_static_score[index_static] 

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

    final_prefix = final_proposals[:, 0].clone().detach()
    #### dynamic scoring #####

    return [final_proposals], [final_proposals_scores], [final_prefix], num_frames


def localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu=False):
    answer = []
    for query in query_json:
        proposals, scores, pre_proposals, num_frames = generate_proposal_revise(video_feature, query['descriptions'], stride, hyperparams, kmeans_gpu)
        
        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = proposals[0] / num_frames * duration
            dynamic_pred = pre_proposals[0] / num_frames * duration
            scores = scores[0]
            scores = scores / scores.max()

        query['response'] = []
        for i in range(len(static_pred)):
            query['response'].append({
                'start': float(dynamic_pred[i]),
                'static_start': float(static_pred[i][0]),
                'end': float(static_pred[i][1]),
                'confidence': float(scores[i])
            })
        answer.append(query)

    proposals = []
    cand_num = hyperparams['cand_num']
    for t in range(cand_num):
        proposals += [[p['response'][t]['static_start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answer if len(p['response']) > t]  ### only static
    
    return proposals