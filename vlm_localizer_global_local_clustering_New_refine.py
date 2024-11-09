import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from llm_prompting import select_proposal
from collections import defaultdict

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
# model, vis_processors, text_processors = load_model_and_preprocess("clip_feature_extractor", model_type="ViT-L-14", device='cuda', is_eval=True) ### CLIP

vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])


def nms(moments, scores, pre_mom, pre_score, thresh):
    scores = scores + pre_score * 0.0
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    pre_mom = pre_mom[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True
    return moments[~suppressed], pre_mom[~suppressed], scores[~suppressed]


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


from sklearn.cluster import KMeans


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

def calc_scores(video_features, sentences, gt, duration):
    num_frames = video_features.shape[0]
    gt = torch.round(torch.tensor(gt) / torch.tensor(duration) * num_frames).to(torch.int)
    with torch.no_grad():
        # print(sentences)
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
            'cuda')
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
    v1 = F.normalize(text_feat, dim=-1)
    v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
    scores = torch.einsum('md,npd->mnp', v1, v2)
    scores, scores_idx = scores.max(dim=-1)
    scores = scores.mean(dim=0, keepdim=True)
    ##### scores normalization
    # scores = (scores - scores.min()) / (scores.max() - scores.min())
    ##### scores normalization

    cum_scores = torch.cumsum(scores, dim=1)[0]

    scores_idx = scores_idx.reshape(-1)
    video_features = torch.tensor(video_features).cuda()
    selected_video_features = video_features[torch.arange(num_frames), scores_idx]
    time_features = (torch.arange(num_frames) / num_frames).unsqueeze(1).cuda()
    selected_video_time_features = torch.cat((selected_video_features, time_features), dim=1)

    #### 비디오 프레임 벡터 스무딩 (글로벌)
    smooth_kernel_size = 21
    smooth_padding = smooth_kernel_size // 2
    padding_selected_video_time_features_global = torch.cat((selected_video_time_features[0].repeat(smooth_padding, 1),
                                                      selected_video_time_features,
                                                      selected_video_time_features[-1].repeat(smooth_padding, 1)),
                                                     dim=0)
    kernel = torch.ones(padding_selected_video_time_features_global.shape[1], 1, smooth_kernel_size).cuda() / smooth_kernel_size
    padding_selected_video_time_features_global = padding_selected_video_time_features_global.unsqueeze(0).permute(0, 2, 1)  # (1, 257, 104)
    smoothed_selected_video_time_features_global = F.conv1d(padding_selected_video_time_features_global, kernel, padding=0,
                                                     groups=padding_selected_video_time_features_global.shape[1])
    smoothed_selected_video_time_features_global = smoothed_selected_video_time_features_global.permute(0, 2, 1)
    selected_video_time_features_global = smoothed_selected_video_time_features_global[0]
    #### 비디오 프레임 벡터 스무딩 (글로벌)

    #### 비디오 프레임 벡터 스무딩 (로컬)
    smooth_kernel_size = 10
    smooth_padding = smooth_kernel_size // 2
    padding_selected_video_time_features_local = torch.cat((selected_video_time_features[0].repeat(smooth_padding, 1),
                                                      selected_video_time_features,
                                                      selected_video_time_features[-1].repeat(smooth_padding, 1)),
                                                     dim=0)
    kernel = torch.ones(padding_selected_video_time_features_local.shape[1], 1,
                        smooth_kernel_size).cuda() / smooth_kernel_size
    padding_selected_video_time_features_local = padding_selected_video_time_features_local.unsqueeze(0).permute(0, 2, 1)  # (1, 257, 104)
    smoothed_selected_video_time_features_local = F.conv1d(padding_selected_video_time_features_local, kernel, padding=0,
                                                     groups=padding_selected_video_time_features_local.shape[1])
    smoothed_selected_video_time_features_local = smoothed_selected_video_time_features_local.permute(0, 2, 1)
    selected_video_time_features_local = smoothed_selected_video_time_features_local[0]
    selected_video_time_features_local_v2 = F.normalize(selected_video_time_features_local[:,:256], dim=1)

    #### 비디오 프레임 벡터 스무딩 (로컬)
    
    #### K-means 클러스터링 적용 (글로벌)
    kmeans_k = 4  ### 4
    if num_frames < kmeans_k:
        kmeans_k = 2
    kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=42)
    kmeans_labels_global = kmeans.fit_predict(np.array(selected_video_time_features_global.cpu()))
    kmeans_labels_global = torch.tensor(kmeans_labels_global)
    #### K-means 클러스터링 적용 (글로벌)

    #### K-means 클러스터링 적용 (로컬)
    kmeans_k = 4  ### 4
    if num_frames < kmeans_k:
        kmeans_k = 2
    kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=42)
    kmeans_labels_local = kmeans.fit_predict(np.array(selected_video_time_features_local.cpu()))
    kmeans_labels_local = torch.tensor(kmeans_labels_local)
    #### K-means 클러스터링 적용 (로컬)

    # 각 클러스터의 요소를 저장할 리스트를 초기화
    cluster_elements = [[] for _ in range(kmeans_k)]
    interval = [0] 
    interval_label = kmeans_labels_global[0].item()

    for idx, label in enumerate(kmeans_labels_global[1:], start=1):  # 첫 번째 요소 이후부터 순회
        if label.item() != interval_label:
            # 현재 구간을 클러스터에 저장하고 새 구간 시작
            cluster_elements[interval_label].append(interval)
            interval = [idx]
            interval_label = label.item()
        else:
            interval.append(idx)
        
    cluster_elements[interval_label].append(interval)
    
    cluster_static_scores = [0 for _ in range(kmeans_k)]
    for cluster_idx in range(kmeans_k):
        internal_indices = torch.tensor([i for interval in cluster_elements[cluster_idx] for i in interval], device='cuda')
        external_indices = torch.tensor([i for i in range(num_frames) if i not in internal_indices], device='cuda')
        cluster_inner_scores = scores[0, internal_indices]
        cluster_inner_mean = cluster_inner_scores.mean()
        
        # 클러스터 외부 요소의 scores 합과 평균
        cluster_outer_scores = scores[0, external_indices]
        cluster_outer_mean = cluster_outer_scores.mean()
        cluster_static_scores[cluster_idx] = cluster_inner_mean - cluster_outer_mean

    # 가장 큰 cluster_static_score를 가진 클러스터 찾기
    max_cluster_idx = torch.argmax(torch.tensor(cluster_static_scores)).item()
    # internal_indices = sorted([i for interval in cluster_elements[max_cluster_idx] for i in interval])
    # new_start = internal_indices[0]
    # new_end = internal_indices[-1]
    # new_kmeans_k = len(cluster_elements[max_cluster_idx])
    # import pdb;pdb.set_trace()


    # 해당 클러스터의 내부 요소 인덱스 가져오기
    internal_indices = sorted([i for interval in cluster_elements[max_cluster_idx] for i in interval])

    # 비연속적인 경계 찾기
    boundaries = []
    start = internal_indices[0]

    for i in range(1, len(internal_indices)):
        if internal_indices[i] != internal_indices[i - 1] + 1:  # 연속되지 않는 경우 경계로 간주
            boundaries.append([start, internal_indices[i - 1]])  # 연속 구간의 시작과 끝 추가
            start = internal_indices[i]  # 새로운 구간 시작

    # 마지막 구간 추가
    boundaries.append([start, internal_indices[-1]])


    combined_proposals = []
    combined_proposals_static_score = []
    # boundaries에 있는 시작-끝 구간 조합 생성
    for i in range(len(boundaries)):
        start = boundaries[i][0]  # i번째 구간의 시작점
        for j in range(i, len(boundaries)):
            end = boundaries[j][1]  # j번째 구간의 끝점
            if start < end:  # 시작점이 끝점보다 앞에 있는 경우에만 추가
                combined_proposals.append([start, end])
                static_score = extract_static_score(start, end, cum_scores, num_frames, scores).item()
                combined_proposals_static_score.append(static_score)
    
    combined_proposals = torch.tensor(combined_proposals)
    combined_proposals_static_score = torch.tensor(combined_proposals_static_score)
    _, index_static = combined_proposals_static_score.sort(descending=True)
    combined_proposals = combined_proposals[index_static]

    return scores, combined_proposals[:5], combined_proposals_static_score[:5]
    # return scores, final_proposals[:5], final_proposals_scores[:5], local_proposals, local_proposals_scores ############### 잘되는놈 공사 이전


###################### 다른 proposals를 제거한 outer_avg 구하기
def extract_static_score_reject_other_proposal(inner_mask, outer_mask, scores):
    if outer_mask.sum() != 0:
        final_score = (inner_mask * scores).sum() / inner_mask.sum() - (outer_mask * scores).sum() / outer_mask.sum()
    else:
        final_score = (inner_mask * scores).sum() / inner_mask.sum()
    return final_score
###################### 다른 proposals를 제거한 outer_avg 구하기


###################### 중심부로부터 멀어질수록 weight를 적게 주는 gaussian 추가하기
def extract_static_gaussian_score(start, end, cum_scores, num_frames, scores):
    kernel_size = end - start
    if start == 0:
        inner_sum = cum_scores[end - 1]
        # inner_sum = scores[:end-1].sum()
    else:
        inner_sum = cum_scores[end - 1] - cum_scores[start - 1]
        # inner_sum = scores[start:end-1].sum()
    outer_sum = cum_scores[num_frames - 1] - inner_sum

    if kernel_size != num_frames:
        static_score = inner_sum / kernel_size - outer_sum / (num_frames - kernel_size)
    else:
        static_score = inner_sum / kernel_size - (scores[0][0] + scores[0][-1]) / 2  #### 임시방편 느낌
        # static_score = inner_sum / kernel_size
    return static_score
###################### 중심부로부터 멀어질수록 weight를 적게 주는 gaussian 추가하기


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
        static_score = inner_sum / kernel_size - (scores[0][0] + scores[0][-1] / 2)  #### 임시방편 느낌
        # static_score = inner_sum / kernel_size
    return static_score


def extract_avg_score(start, end, cum_scores, num_frames, scores):
    kernel_size = end - start
    if start == 0:
        inner_sum = cum_scores[end - 1]
    else:
        inner_sum = cum_scores[end - 1] - cum_scores[start - 1]
    avg_score = inner_sum / kernel_size
    return avg_score


def generate_proposal(video_features, sentences, gt, duration, stride, max_stride, nms_thresh=0.3):
    num_frames = video_features.shape[0]
    ground_truth = [round(gt[0] / duration * num_frames, 0), round(gt[1] / duration * num_frames, 0)]
    scores, final_proposals, final_proposals_scores = calc_scores(video_features, sentences, gt, duration)
    cum_scores = torch.cumsum(scores, dim=1)[0]

    masks = (scores > 0.2).float()
    scores = scores * masks
    stride = min(stride, scores.size(-1) // 2)
    # dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)

    final_proposals = final_proposals.clone()
    final_scores = final_proposals_scores.clone()
    final_prefix = final_proposals[:, 0].clone().detach()
    final_scores, sort_idx = final_scores.sort(descending=True)
    final_proposals = final_proposals[sort_idx]
    final_prefix = final_prefix[sort_idx]
    ####### important debug
    # print(final_proposals.cpu()*num_frames, ground_truth, final_scores, num_frames) ### frame 단위로 통일
    ####### important debug
    return [final_proposals], [final_scores], [final_prefix], scores, cum_scores, num_frames


def localize(video_feature, duration, query_json, stride, max_stride):
    answer = []
    for query in query_json:
        # import pdb; pdb.set_trace()
        gt = query['gt']
        duration = query['duration']
        proposals, scores, pre_proposals, ori_scores, ori_cum_scores, num_frames = generate_proposal(video_feature, query['descriptions'], gt,
                                                                         duration, stride, max_stride)


        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = proposals[0][:10]
            dynamic_pred = pre_proposals[0][:10]
            # scores = gmm_scores[:10]
            # scores = scores / scores.max()
            scores = scores[0][:10]
            scores = scores / scores.max()


            # if scores.min() < 0:
            #     scores = scores + (-scores.min() + 1e-4)
            # scores = scores / scores.max()
            # scores = scores + (1- scores.max()) #### 공사중

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
    for t in range(5): ##################### 건들여봐야해!!! 성준아
        proposals += [[p['response'][t]['static_start'], p['response'][t]['end'], p['response'][t]['confidence']] for p
                      in answer if len(p['response']) > t]  ### only static
    proposals = np.array(proposals)
    proposals[:,:2] = proposals[:,:2] / num_frames * duration
    post_proposals = proposals
    # print(np.array(proposals)[:3])
    # post_proposals = post_processing(proposals, local_proposals, local_proposals_scores, gt, num_frames, duration, ori_cum_scores, ori_scores) ### Refinement
    # print(post_proposals[:3])
    np.set_printoptions(precision=4, suppress=True)
    # print(post_proposals)
    post_proposals = select_proposal(np.array(post_proposals))
    # print(post_proposals)
    # print(gt, duration)
    # print('===================================')
    return post_proposals

