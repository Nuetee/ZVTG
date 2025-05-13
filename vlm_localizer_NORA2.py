import os
import clip
import torch
import numpy as np
from scipy.optimize import minimize_scalar
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from twfinch import FINCH
from llm_prompting import select_proposal

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


import torch

def temporal_kmeans_clustering_gpu(k, features, frame_indices, n_iter=100, tol=1e-4):
    torch.manual_seed(60)
    device = features.device
    features = features.cuda()
    frame_indices = frame_indices.cuda()
    n_samples, n_features = features.shape

    # --- KMeans++ 초기화 ---
    centroids = torch.empty((k, n_features), device=device)
    random_idx = torch.randint(0, n_samples, (1,), device=device)
    centroids[0] = features[random_idx]

    for i in range(1, k):
        dists = torch.cdist(features, centroids[:i]) ** 2
        min_dists, _ = torch.min(dists, dim=1)
        probs = min_dists / min_dists.sum()
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=device)
        next_idx = torch.searchsorted(cumulative_probs, r).item()
        centroids[i] = features[next_idx]

    # --- 클러스터링 반복 ---
    labels = torch.full((n_samples,), -1, dtype=torch.long, device=device)
    tolerances = [1, 3, 5]

    for tol_frame in tolerances:
        for _ in range(n_iter):
            dists = torch.cdist(features, centroids, p=2)
            changed = False
            new_labels = labels.clone()

            for i in range(n_samples):
                if labels[i] != -1:
                    continue

                nearest_cluster = torch.argmin(dists[i])
                assigned_idx = (labels == nearest_cluster).nonzero(as_tuple=True)[0]

                if assigned_idx.numel() == 0:
                    if any(torch.abs(frame_indices[i] - frame_indices[j]) <= tol_frame for j in range(n_samples) if labels[j] == -1):
                        new_labels[i] = nearest_cluster
                        changed = True
                else:
                    if torch.any(torch.abs(frame_indices[assigned_idx] - frame_indices[i]) <= tol_frame):
                        new_labels[i] = nearest_cluster
                        changed = True

            labels = new_labels

            # 중심점 업데이트
            new_centroids = torch.stack([
                features[labels == j].mean(dim=0) if (labels == j).sum() > 0 else centroids[j]
                for j in range(k)
            ])
            if torch.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids

        if (labels == -1).sum() == 0:
            break
    
    # --- 클러스터 정리: 1개 샘플만 있는 클러스터 제거 ---
    final_labels = labels.clone()
    
    for cluster_id in range(k):
        cluster_sample_indices = (final_labels == cluster_id).nonzero(as_tuple=True)[0]
        if cluster_sample_indices.numel() == 1:
            final_labels[cluster_sample_indices] = -1  # 미할당 처리
            
    # --- 후처리: 연속 5프레임 클러스터 + 나머지 병합 ---
    unassigned = (final_labels == -1).nonzero(as_tuple=True)[0]

    if unassigned.numel() > 0:
        sorted_idx = unassigned[torch.argsort(frame_indices[unassigned])]
        group = []
        temp_cluster_id = k
        assigned_noise = []

        for idx in sorted_idx:
            if not group or frame_indices[idx] == frame_indices[group[-1]] + 1:
                group.append(idx.item())
            else:
                if len(group) >= 5:
                    final_labels[group] = temp_cluster_id
                    temp_cluster_id += 1
                else:
                    assigned_noise.extend(group)
                group = [idx.item()]

        if len(group) >= 5:
            final_labels[group] = temp_cluster_id
        else:
            assigned_noise.extend(group)

        # 노이즈 샘플 → 시간적으로 인접하고 가장 가까운 클러스터로 할당
        for idx in assigned_noise:
            candidate_clusters = []
            for cluster_id in range(k):
                cluster_indices = (final_labels == cluster_id).nonzero(as_tuple=True)[0]
                if cluster_indices.numel() == 0:
                    continue
                if torch.any(torch.abs(frame_indices[cluster_indices] - frame_indices[idx]) <= 5):
                    candidate_clusters.append(cluster_id)

            if candidate_clusters:
                dists_to_clusters = torch.stack([
                    torch.norm(features[idx] - centroids[cid]) for cid in candidate_clusters
                ])
                nearest = candidate_clusters[torch.argmin(dists_to_clusters)]
                final_labels[idx] = nearest

    return final_labels.cpu()


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


def generate_proposal_revise(video_features, sentences, stride, hyperparams, kmeans_gpu):
    num_frames = video_features.shape[0]

    with torch.no_grad():
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
    v1 = F.normalize(text_feat, dim=-1)
    v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
    scores = torch.einsum('md,npd->mnp', v1, v2)
    scores, scores_idx = scores.max(dim=-1)
    scores = scores.mean(dim=0, keepdim=True)

    initial_masks = (scores > 0.2 if hyperparams['is_blip2'] else scores > 0)
    masks, masked_indices = scores_masking(scores, initial_masks)

    data = scores[:, masks].flatten().cpu().numpy()
    normalized_scores, is_scale = alignment_adjustment(data, hyperparams['gamma'], scores.device, lambda_max=2, lambda_min=-2)

    if hyperparams['is_blip2'] or hyperparams['is_blip']:
        video_features = torch.tensor(video_features).cuda()
        scores_idx = scores_idx.reshape(-1)
        selected_video_features = video_features[torch.arange(num_frames), scores_idx]
    else:
        selected_video_features = video_features

    time_features = (torch.arange(num_frames) / num_frames).unsqueeze(1).cuda()
    selected_video_time_features = torch.cat((selected_video_features, time_features), dim=1)
    selected_video_time_features = selected_video_time_features[masks]

    temporal_aware_features = temporal_aware_feature_smoothing(hyperparams['temporal_window_size'], selected_video_time_features)

    kmeans_k = min(hyperparams['kmeans_k'], max(2, len(masked_indices)))
    cluster_labels = kmeans_clustering_gpu(kmeans_k, temporal_aware_features)
    
    def compute_cluster_scores(cluster_labels, similarity_scores):
        cluster_scores = {}
        for cluster_id in torch.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_scores[cluster_id.item()] = similarity_scores[0][masks][cluster_mask].mean().item()
        return cluster_scores

    def get_proposals_with_cluster_scores(cluster_labels, cluster_scores, num_frames, prior, masked_indices):
        proposals = []
        proposal_scores = []
        for i in range(num_frames):
            for j in range(i + 1, num_frames):
                if (j - i) > num_frames * prior:
                    continue
                inner_mask = (masked_indices >= i) & (masked_indices < j)
                outer_mask = (masked_indices < i) | (masked_indices >= j)

                inner_clusters = torch.unique(cluster_labels[inner_mask])
                outer_clusters = torch.unique(cluster_labels[outer_mask])
                
                if len(inner_clusters) == 0 or len(outer_clusters) == 0:
                    continue

                inner_score = torch.tensor([cluster_scores[c.item()] for c in inner_clusters]).mean()
                outer_score = torch.tensor([cluster_scores[c.item()] for c in outer_clusters]).mean()
                proposals.append([i, j])
                proposal_scores.append((inner_score - outer_score).item())
        return proposals, proposal_scores


    cluster_scores = compute_cluster_scores(cluster_labels, normalized_scores)
    masked_indices_cpu = masked_indices.cpu()
    cluster_labels_cpu = cluster_labels.cpu()

    final_proposals, final_scores = get_proposals_with_cluster_scores(cluster_labels_cpu, cluster_scores, num_frames, hyperparams['prior'], masked_indices_cpu)


    final_proposals = torch.tensor(final_proposals)
    final_scores = torch.tensor(final_scores)
    _, sorted_idx = final_scores.sort(descending=True)
    final_proposals = final_proposals[sorted_idx]
    final_scores = final_scores[sorted_idx]

    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores * initial_masks.float(), stride, initial_masks.float())
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

    return [final_proposals], [final_scores], [final_prefix], num_frames


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
        proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answer if len(p['response']) > t]  ### only static
    
    return proposals