import os
import clip
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


#### CLIP ####
clip_model, preprocess = clip.load("ViT-L/14", device='cuda')
clip_text_encoder = clip_model.encode_text
#### CLIP ####

#### InternVideo ####
import sys
from pathlib import Path
# InternVideo 모듈이 있는 디렉토리를 sys.path에 추가
module_path = Path("./InternVideo/InternVideo1/Pretrain/Multi-Modalities-Pretraining").resolve()
if str(module_path) not in sys.path:
    sys.path.append(str(module_path))

import InternVideo
model_internVideo = InternVideo.load_model("./InternVideo/InternVideo1/Pretrain/Multi-Modalities-Pretraining/InternVideo-MM-L-14.ckpt").cuda()
#### InternVideo ####

def iou(candidates, gt):
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def nms(video_segments, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter video segments based on their scores and overlap.

    Parameters:
        video_segments (torch.Tensor): Tensor of shape (N, 2) containing the start and end times of video segments.
        scores (torch.Tensor): Tensor of shape (N,) containing the scores for each segment.
        iou_threshold (float): IOU threshold for suppression.

    Returns:
        torch.Tensor: Indices of the segments that are kept after NMS.
    """
    if len(video_segments) == 0:
        return torch.tensor([], dtype=torch.long)

    # Sort segments by their scores in descending order
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(indices) > 0:
        # Select the segment with the highest score
        current = indices[0].item()
        keep.append(current)

        # Compute IoU with the remaining segments
        iou_list = []
        for idx in indices[1:]:
            iou_list.append(compute_iou(video_segments[current], video_segments[idx]))
        
        iou_list = torch.tensor(iou_list)

        # Filter out segments with IoU above the threshold
        indices = indices[1:][iou_list <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)

def compute_iou(segment_a, segment_b):
    """
    Compute Intersection over Union (IoU) between two segments.

    Parameters:
        segment_a (torch.Tensor): Start and end times of the first segment.
        segment_b (torch.Tensor): Start and end times of the second segment.

    Returns:
        float: IoU value.
    """
    start_a, end_a = segment_a
    start_b, end_b = segment_b

    # Compute the intersection
    intersection_start = max(start_a.item(), start_b.item())
    intersection_end = min(end_a.item(), end_b.item())
    intersection = max(0, intersection_end - intersection_start)

    # Compute the union
    union = (end_a.item() - start_a.item()) + (end_b.item() - start_b.item()) - intersection

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union


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


# calc_scores 함수 실행 후에 scores와 normalized_scores를 입력으로 사용합니다.
import matplotlib.pyplot as plt
def plot_scores(scores, normalized_scores, timestamps, filename="scores_plot.png"):
    # scores와 normalized_scores를 GPU에서 CPU로 이동시키고 numpy 배열로 변환
    scores_np = scores.squeeze().cpu().numpy()
    normalized_scores_np = normalized_scores.squeeze().cpu().numpy()

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(scores_np, label="Scores", linestyle="-")
    plt.plot(normalized_scores_np, label="Normalized Scores", linestyle="-")
    plt.xlabel("Frame Index")
    plt.ylabel("Score")
    plt.title("Scores and Normalized Scores")
    plt.legend()
    plt.grid(True)
    
    # timestamps 구간을 회색으로 칠하기
    start, end = timestamps 
    plt.axvspan(start, end, color='gray', alpha=0.3)  # 회색 구간 추가

    # 그래프를 파일로 저장
    plt.savefig(filename)
    plt.close()  # 메모리 절약을 위해 그래프 닫기


def feature_tsne(features, sentence, gt, save_dir):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    # from adjustText import adjust_text

    # Normalize features
    normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
    normalized_features_np = normalized_features.detach().cpu().numpy()
    n_samples = normalized_features_np.shape[0]
    perplexity = min(30, n_samples - 1)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_features = tsne.fit_transform(normalized_features_np)

    # Plot t-SNE with indices, using red color for points within the timestamp range
    start = gt[0]
    end = gt[1]
    plt.figure(figsize=(12, 8))

    # 범례에 사용할 빈 점 추가
    plt.scatter([], [], c='red', label='Ground Truth Segment', s=35)
    plt.scatter([], [], c='blue', label='Non-Ground Truth Segment', s=35)

    # texts = []
    for i, (x, y) in enumerate(tsne_features):
        color = 'red' if start <= i <= end else 'blue'
        plt.scatter(x, y, c=color, s=35)
        # if start <= i <= end:
        #     text = plt.text(x, y, str(i), fontsize=10, ha='center')  # Add text only for gt segment
        #     texts.append(text)

    # Adjust text to prevent overlap
    # adjust_text(texts, only_move={'points': 'y', 'texts': 'y'}, autoalign='y', force_text=0.5)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=18, loc="lower right")
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/tsne_features_{sentence}.png")


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

    return kmeans_labels, kmeans.inertia_


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

    # WCSS 계산
    distances = torch.cdist(features, centroids, p=2)
    min_distances = distances.gather(1, labels.unsqueeze(1)).squeeze()
    wcss = (min_distances**2).sum().item()  # WCSS 값 반환

    return labels.cpu(), wcss

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


def generate_proposal_revise(video_features, sentences, stride, kmeans_k, hyperparams, kmeans_gpu):
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
    elif hyperparams['is_internVideo']:
        with torch.no_grad():
            text = InternVideo.tokenize([sentences]).cuda()
            text_feat =  model_internVideo.encode_text(text)
            video_features = torch.tensor(video_features).cuda()

            v1 = F.normalize(text_feat, dim=1)
            v2 = torch.nn.functional.normalize(video_features, dim=1)
            scores = (v1 @ v2.T) # (num_segments, num_texts)
    else:
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

    # Temporal-aware vector smoothing
    temporal_aware_features = temporal_aware_feature_smoothing(hyperparams['temporal_window_size'], selected_video_time_features)
       
    from kneed import KneeLocator 
    # Kmeans Clustering
    k_values = range(3, 20)  # 원래 설정된 k 범위
    actual_k_values = []  # 실제 사용된 k 저장
    wcss_values = []
    last_actual_k = None  # 마지막으로 추가된 actual_k 값 추적

    for kmeans_k in k_values:
        actual_k = min(kmeans_k, max(2, len(masked_indices)))  # 실제 적용된 k
        
        # 이전과 같은 actual_k라면 중복 제거
        if actual_k == last_actual_k:
            continue

        last_actual_k = actual_k  # 중복 방지를 위해 현재 k 저장
        actual_k_values.append(actual_k)
        
        if kmeans_gpu:
            _, wcss = kmeans_clustering_gpu(actual_k, temporal_aware_features)
        else:
            _, wcss = kmeans_clustering(actual_k, temporal_aware_features)

        wcss_values.append(wcss)

    if len(actual_k_values) > 1:
        # KneeLocator 적용 (실제 사용된 k값을 기준으로)
        kneedle = KneeLocator(actual_k_values, wcss_values, curve="convex", direction="decreasing")
        optimal_k = kneedle.elbow
    else:
        optimal_k = actual_k_values[0]

    if optimal_k is None:
       optimal_k = max(2, max(actual_k_values))
    
    if kmeans_gpu:
        kmeans_labels, _ = kmeans_clustering_gpu(optimal_k, temporal_aware_features)
    else:
        kmeans_labels, _ = kmeans_clustering(optimal_k, temporal_aware_features)


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


def localize(video_feature, duration, query_json, stride, kmeans_k, hyperparams, kmeans_gpu=False):
    answer = []
    for query in query_json:
        proposals, scores, pre_proposals, num_frames = generate_proposal_revise(video_feature, query['descriptions'], stride, kmeans_k, hyperparams, kmeans_gpu)
        
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