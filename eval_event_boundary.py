import argparse
import torch
import os
import json
import numpy as np
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from sklearn.cluster import KMeans
from tqdm import tqdm
from data_configs import DATASETS

#### BLIP-2 Q-Former ####
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])
#### BLIP-2 Q-Former ####

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


def generate_proposal_revise(video_features, sentences, stride, hyperparams, kmeans_gpu):
    num_frames = video_features.shape[0]

    with torch.no_grad():
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
            'cuda')
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
    v1 = F.normalize(text_feat, dim=-1)
    v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
    scores = torch.einsum('md,npd->mnp', v1, v2)
    scores, scores_idx = scores.max(dim=-1)
    scores = scores.mean(dim=0, keepdim=True)
    
    # scores > 0.2인 마스킹 생성 (Boolean 형태 유지)
    initial_masks = (scores > 0.2 if hyperparams['is_blip2'] else scores > 0)
    masks, masked_indices = scores_masking(scores, initial_masks)

    # Alignment adjustment of similarity scores
    data = scores[:, masks].flatten().cpu().numpy()   # 마스크된 부분만 가져오기    
    
    video_features = torch.tensor(video_features).cuda()
    scores_idx = scores_idx.reshape(-1)
    selected_video_features = video_features[torch.arange(num_frames), scores_idx]
        
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

    

    return scene_segments


def localize(video_feature, duration, gt, query_json, stride, hyperparams, kmeans_gpu=False):
    num_frames = video_feature.shape[0]
    boundaries = [timestamp[0] for timestamp in gt]
    if 0 not in boundaries:
        boundaries.append(0)
    boundaries.sort()
    boundaries = [(boundary / duration) * num_frames for boundary in boundaries]

    for query in query_json:
        scene_segments = generate_proposal_revise(video_feature, query['descriptions'], stride, hyperparams, kmeans_gpu)
        

    return scene_segments

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use only VLM for evaluation.')

def eval_TAG(data, feature_path, stride, hyperparams, kmeans_gpu):

    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        
        for i in range(len(ann['sentences'])):
            gt = ann['timestamps']
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, gt, query_json, stride, hyperparams, kmeans_gpu)