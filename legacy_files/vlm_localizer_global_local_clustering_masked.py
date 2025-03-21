import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from llm_prompting import select_proposal

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

def calc_scores_with_indices(video_features, sentences, indices):
    with torch.no_grad():
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
              
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:,0,:])
    
    v1 = F.normalize(text_feat, dim=-1)
    v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
    # 텍스트와 비디오 특징 간의 내적(유사도) 계산
    scores = torch.einsum('md,npd->mnp', v1, v2)
    scores = scores.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    scores = scores.mean(dim=0, keepdim=True)

    return scores

def calc_scores_masked(video_features, sentences, masked_sentences, gt, duration, high=1.1, low=0.7):
    num_frames = video_features.shape[0]
    gt = torch.round(torch.tensor(gt) / torch.tensor(duration) * num_frames).to(torch.int)
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

    cum_scores = torch.cumsum(scores, dim=1)[0]

    # masked importance scores calcualtion
    masked_query_scores = []
    for masked_element, masked_queries in masked_sentences.items():
        if "prepositional" in masked_element or "subject" in masked_element or "object" in masked_element:
            continue
        for masked_query in masked_queries:
            if len(masked_query) == 0:
                continue
            masked_query_score = calc_scores_with_indices(video_features, [masked_query], scores_idx)
            masked_query_scores.append(masked_query_score)
   
    importance_scores_list = []
    for masked_query_score in masked_query_scores:
        importance_scores = 1 - masked_query_score / scores
        importance_scores_list.append(importance_scores)
    if len(masked_query_scores) == 0:
        importance_scores = torch.ones_like(scores)
    else:
        importance_scores_tensor = torch.stack(importance_scores_list, dim=0)
        importance_scores =  torch.amax(importance_scores_tensor, dim=0)
        # importance_scores = importance_scores_tensor.mean(dim=0) # 2D 텐서로 변환
    cum_importance_scores = torch.cumsum(importance_scores, dim=1)[0]
    # masked importance scores calcualtion

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
    smooth_kernel_size = 21
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

    #### (글로벌) 클러스터링 결과에 따라 묶음 만들기
    global_proposals = []
    global_proposals_scores = []
    start_idx = 0
    current_val = kmeans_labels_global[0]
    for i in range(1, num_frames):
        if kmeans_labels_global[i] != current_val:
            global_proposals.append([start_idx, i])  ### start_idx 이상, i 미만 까지 같은 레이블
            score = extract_static_score(start_idx, i, cum_scores, num_frames, scores).item()
            global_proposals_scores.append(round(score, 4))
            start_idx = i
            current_val = kmeans_labels_global[i]
    global_proposals.append([start_idx, num_frames])
    score = extract_static_score(start_idx, i, cum_scores, num_frames, scores).item()
    global_proposals_scores.append(round(score, 4))
    global_proposals.append([num_frames, num_frames])
    #### (글로벌) 클러스터링 결과에 따라 묶음 만들기

    #### (로컬) 클러스터링 결과에 따라 묶음 만들기
    local_proposals = []
    local_proposals_scores = []
    start_idx = 0
    current_val = kmeans_labels_local[0]
    for i in range(1, num_frames):
        if kmeans_labels_local[i] != current_val:
            local_proposals.append([start_idx, i])  ### start_idx 이상, i 미만 까지 같은 레이블
            score = extract_static_score(start_idx, i, cum_scores, num_frames, scores).item()
            local_proposals_scores.append(round(score, 4))
            start_idx = i
            current_val = kmeans_labels_local[i]
    local_proposals.append([start_idx, num_frames])
    score = extract_static_score(start_idx, i, cum_scores, num_frames, scores).item()
    local_proposals_scores.append(round(score, 4))
    local_proposals.append([num_frames, num_frames])
    #### (로컬) 클러스터링 결과에 따라 묶음 만들기
    
    ### (글로벌) Extracting Global Proposals (Cartesian product)
    final_proposals = []
    final_proposals_scores_static = []
    final_proposals_importance_scores = []
    for i in range(len(global_proposals)):
        for j in range(i + 1, len(global_proposals)):
            start = global_proposals[i][0]
            last = global_proposals[j][0]
            # if (last - start) > num_frames * 0.5:
            #     continue
            score_static = extract_static_score(start, last, cum_scores, num_frames, scores).item()
            importance_score = extract_static_score(start, last, cum_importance_scores, num_frames, scores).item()

            final_proposals.append([start, last])
            final_proposals_scores_static.append(round(score_static, 4))
            final_proposals_importance_scores.append(round(importance_score, 4))
    
    final_proposals_scores_static = torch.tensor(final_proposals_scores_static)
    final_proposals_importance_scores = torch.tensor(final_proposals_importance_scores)
    
    # temperature = 1 / len(final_proposals)
    # final_proposals_scores_static = torch.nn.functional.softmax(final_proposals_scores_static / 0.01, dim=0)
    
    # # 최소-최대 정규화로 범위를 조정
    # min_val = torch.min(final_proposals_importance_scores)
    # max_val = torch.max(final_proposals_importance_scores)
    # normalized_importance_scores = (final_proposals_importance_scores - min_val) / (max_val - min_val)
    
    # # flattened_scores의 범위에 맞추어 스케일링
    # scores_min = torch.min(final_proposals_scores_static)
    # scores_max = torch.max(final_proposals_scores_static)
    # scaled_importance_scores = normalized_importance_scores * (scores_max - scores_min) + scores_min
    for idx, importance_score in enumerate(final_proposals_importance_scores):
        if importance_score > 0:
            final_proposals_scores_static[idx] *= high
        else:
            final_proposals_scores_static[idx] *= low
    
    # 조정된 값을 소프트맥스에 적용
    # final_proposals_importance_scores = torch.nn.functional.softmax(scaled_importance_scores / temperature, dim=0)
    # final_proposals_scores_static = final_proposals_scores_static * (final_proposals_importance_scores**gamma)
    
    final_proposals = torch.tensor(final_proposals)
    value_static, index_static = final_proposals_scores_static.sort(descending=True)
    final_proposals_static = final_proposals[index_static]
    final_proposals_scores_static = final_proposals_scores_static[index_static]
    final_proposals_importance_scores = final_proposals_importance_scores[index_static]


    return scores, final_proposals_static[:5], final_proposals_scores_static[:5], local_proposals, local_proposals_scores


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


def generate_proposal_masked(video_features, sentences, masked_sentences, gt, duration, stride, high=1.1, low=0.7):
    num_frames = video_features.shape[0]
    scores, final_proposals, final_proposals_scores, local_proposals, local_proposals_scores = calc_scores_masked(video_features, sentences, masked_sentences, gt, duration, high=high, low=low)
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
    
    return [final_proposals], [final_scores], [final_prefix], scores, cum_scores, local_proposals, local_proposals_scores, num_frames


def localize(video_feature, duration, query_json, stride, max_stride, high=1.1, low=0.7):
    answer = []
    for query in query_json:
        # import pdb; pdb.set_trace()
        gt = query['gt']
        duration = query['duration']
        proposals, scores, pre_proposals, _, _, _, _, num_frames = generate_proposal_masked(video_feature, query['descriptions'], query['masked_descriptions'], gt, duration, stride, high=high, low=low)

        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = proposals[0][:10]
            dynamic_pred = pre_proposals[0][:10]
            scores = scores[0][:10]
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
    for t in range(5): ##################### 건들여봐야해!!! 성준아
        proposals += [[p['response'][t]['static_start'], p['response'][t]['end'], p['response'][t]['confidence']] for p
                      in answer if len(p['response']) > t]  ### only static
    proposals = np.array(proposals)
    proposals[:,:2] = proposals[:,:2] / num_frames * duration
    post_proposals = proposals

    post_proposals = select_proposal(np.array(post_proposals))
    return post_proposals
