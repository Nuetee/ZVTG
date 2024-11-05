import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms


model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda', is_eval=True)
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

    # 높은 static 점수를 갖는 순으로 구간을 정렬하고 앞뒤로 겹치는 구간이 큰 경우 해당 구간의 suppressed(억제)를 True로
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
    return moments[~suppressed], pre_mom[~suppressed], scores[~suppressed]


def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def get_dynamic_scores(scores, stride, masks, ths=0.0005, sigma=1):
    def gaussian_kernel(size, sigma=1):
        size = int(size) // 2
        x = np.arange(-size, size+1)
        normal = 1 / (np.sqrt(2.0 * np.pi) * sigma)
        g =  np.exp(-x**2 / (2.0 * sigma**2)) * normal
        return g
    
    # 중요조건 3개: 현재 구간의 미분값이 ths 이상 / 현재 구간 미분값과 직전구간 미분값의 조합이 ths 이상 / 현재 구간, 직전 구간, 두번째 이전 구간의 미분값 조합이 ths 이상
    def nchk(f, f1, f2, ths):
        return (((3 * f) > ths) | ((2 * f + f1) > ths) | ((f + f1 + f2) > ths))
    
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
    diffres = torch.cat((pad_left, diffres, pad_right), dim = -1) * masks

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
            # 변화량에 대한 중요 조건 3개 중 하나라도 만족할 경우, dynamic score를 증가. 그렇지 않으면 dynamic score를 0으로 초기화하고 d_idx를 현재위치로 업데이트
            if nchk(f1, f2, f3, ths):
                d_score += max(3 * f1,2 * f1 + f2,f1 + f2 + f3)
            else:
                d_idx = i
                d_score = 0
            
            # 같은 동적구간은 같은 d_idx를 가지며, 같은 동적구간 내에 증가하는 d_score를 가짐
            dynamic_idxs[idx][i] = d_idx / scores.size(-1)
            dynamic_scores[idx][i] = d_score

    dynamic_idxs = torch.from_numpy(dynamic_idxs).to('cuda')
    dynamic_scores = torch.from_numpy(dynamic_scores).to('cuda')

    return dynamic_idxs, dynamic_scores

def calc_scores(video_features, sentences):
    with torch.no_grad():
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')                    
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:,0,:])
    
    v1 = F.normalize(text_feat, dim=-1)
    v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
    # 텍스트와 비디오 특징 간의 내적(유사도) 계산
    scores = torch.einsum('md,npd->mnp', v1, v2)
    scores, _ = scores.max(dim=-1)
    scores = scores.mean(dim=0, keepdim=True)

    return scores

def calc_scores2(video_features, sentences):
    with torch.no_grad():
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')          
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:,0,:])
    
    v1 = F.normalize(text_feat, dim=-1)
    v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
    # 텍스트와 비디오 특징 간의 내적(유사도) 계산
    scores = torch.einsum('md,npd->mnp', v1, v2)
    scores, indices = scores.max(dim=-1)
    scores = scores.mean(dim=0, keepdim=True)

    return scores, indices

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

def generate_proposal_with_replaced(video_features, sentences, replaced_sentences, stride, max_stride, gamma, nms_thresh=0.3):
    scores, indices = calc_scores2(video_features, sentences)
    masks = (scores > 0.2).float()
    scores = scores * masks
    
    replaced_query_scores = []
    for replaced_element, replaced_queries in replaced_sentences.items():
        if "prepositional" in replaced_element or "subject" in replaced_element or "object" in replaced_element:
            continue
        for replaced_query in replaced_queries:
            if len(replaced_query) == 0:
                continue
            replaced_query_score = calc_scores_with_indices(video_features, [replaced_query], indices)
            replaced_query_scores.append(replaced_query_score)

    importance_scores_list = []
    for replaced_query_score in replaced_query_scores:
        importance_scores = 1 - replaced_query_score / scores
        importance_scores_list.append(importance_scores)
    if len(replaced_query_scores) == 0:
        importance_scores = torch.ones_like(scores)
    else:
        importance_scores_tensor = torch.stack(importance_scores_list, dim=0)
        # importance_scores =  torch.amax(importance_scores_tensor, dim=0)
        importance_scores = importance_scores_tensor.mean(dim=0) # 2D 텐서로 변환


    stride = min(stride, scores.size(-1)//2) # stride가 score 길이 절반을 초과하지 않도록 조정
    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)

    # static scores
    flattened_proposals = [] # 제안 구간
    flattened_scores = [] # 제안 구간에 대한 점수
    flattened_importance_scores = []
    flattened_prefix = [] # 동적 인덱스
    flattened_prescore = [] # 동적 점수
    
    for kernel_size in range(stride, min(scores.size(-1)+1, max_stride+1), stride): # stride에 따라 다양한 크기의 커널을 사용하여 구간 제안을 생성
        kernel = torch.ones((1, 1, kernel_size)).to('cuda') # [1, 1, 20], kernal_size = 20
        inner_sum = F.conv1d(scores.view(-1, 1, scores.size(-1)), kernel).view(scores.size(0), -1) # 커널 내부의 점수 합 (모든 가능 구간에 대한 점수 합)
        inner_num = F.conv1d(masks.view(-1, 1, masks.size(-1)), kernel).view(masks.size(0), -1) # 커널 내부의 유효 요소(0.2 이상의 유사도를 갖는 요소)의 개수
        outer_sum = (scores * masks).sum(dim=-1, keepdim=True) - inner_sum # 커널 외부 구간의 점수 합 (모든 가능 구간에 대한 점수 합)
        outer_num = masks.sum(dim=-1, keepdim=True) - inner_num # 커널 외부의 유효요소 개수
        static_scores = inner_sum / kernel_size - outer_sum / outer_num # 내부 평균 점수 - 외부 평균 점수 (Static scoring) [1,74]

        importance_inner_sum = F.conv1d(importance_scores.view(-1, 1, importance_scores.size(-1)), kernel).view(importance_scores.size(0), -1)
        importance_outer_sum = importance_scores.sum(dim=-1, keepdim=True) - importance_inner_sum
        importance_outer_num = importance_scores.size(-1) - kernel_size
        importance_static_scores = importance_inner_sum / kernel_size - importance_outer_sum / importance_outer_num

        proposals = torch.arange(0, static_scores.size(-1)).to('cuda')
        proposals = torch.stack([proposals, proposals + kernel_size], dim=-1) / scores.size(-1) # 0 ~ 1 정규화 값으로 구간 표현 (ex. [[0, 0.1], [0.1, 0.2], ... [0.9, 1]] -> kernal_size = 0.1), [74, 2]
        
        #### sungjoon ####
        dynamic_idxs_tmp = dynamic_idxs.narrow(-1, 0, static_scores.size(-1)) # [1, 74]
        dynamic_scores_tmp = dynamic_scores.narrow(-1, 0, static_scores.size(-1)) # [1, 74]
        for idx in range(static_scores.size(0)): # static_scores.size(0) = 1
            if idx >= len(flattened_proposals): # for문의 처음 반복
                flattened_proposals.append(proposals)
                flattened_scores.append(static_scores[idx])
                flattened_importance_scores.append(importance_static_scores[idx])
                flattened_prefix.append(dynamic_idxs_tmp[idx])
                flattened_prescore.append(dynamic_scores_tmp[idx])
            else: # for문의 나중 반복
                flattened_proposals[idx] = torch.concat([flattened_proposals[idx], proposals], dim=0)
                flattened_scores[idx] = torch.concat([flattened_scores[idx], static_scores[idx]], dim=0)
                flattened_importance_scores[idx] = torch.concat([flattened_importance_scores[idx], importance_static_scores[idx]])
                flattened_prefix[idx] = torch.concat([flattened_prefix[idx], dynamic_idxs_tmp[idx]], dim=0)
                flattened_prescore[idx] = torch.concat([flattened_prescore[idx], dynamic_scores_tmp[idx]], dim=0)
        #### sungjoon ####
        
        # flattened_proposals는 길이 1짜리 리스트. flattened_proposals[0]에는 계속 proposal들이 append. 처음엔 [74, 2] 두번째 반복엔 [128, 2] ...
    temperature = 1 / len(flattened_scores[0]) # T < 1은 날카로움 증가, T > 1은 평탄함 증가

    scaled_flattened_scores = []
    for scores in flattened_scores:
        softmax_scores = torch.nn.functional.softmax(scores / temperature, dim=0)
        scaled_flattened_scores.append(softmax_scores)

    scaled_flattened_importance_scores = []
    for idx, importance_scores in enumerate(flattened_importance_scores):
        # 최소-최대 정규화로 범위를 조정
        min_val = torch.min(importance_scores)
        max_val = torch.max(importance_scores)
        normalized_importance_scores = (importance_scores - min_val) / (max_val - min_val)
        
        # flattened_scores의 범위에 맞추어 스케일링
        scores = flattened_scores[idx]
        scores_min = torch.min(scores)
        scores_max = torch.max(scores)
        scaled_importance_scores = normalized_importance_scores * (scores_max - scores_min) + scores_min
        
        # 조정된 값을 소프트맥스에 적용
        softmax_importance_scores = torch.nn.functional.softmax(scaled_importance_scores / temperature, dim=0)
        scaled_flattened_importance_scores.append(softmax_importance_scores)

    flattened_scores = [scaled_flattened_scores[0] * (scaled_flattened_importance_scores[0]**gamma)]

    # NMS
    filtered_proposals = []
    filtered_scores = []
    filtered_prefix = []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            # 가능한 모든 proposal(static + dynamic)에 대해, nms
            nms_proposals, nms_prefix, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], flattened_prefix[idx], flattened_prescore[idx], nms_thresh)
            filtered_proposals.append(nms_proposals)
            filtered_scores.append(nms_scores)
            filtered_prefix.append(nms_prefix)
        else:
            filtered_proposals.append([])
            filtered_scores.append([])
            filtered_prefix.append([])

    return filtered_proposals, filtered_scores, filtered_prefix, scores

def localize2(video_feature, duration, query_json, stride, max_stride, gamma):
    answer = []
    for query in query_json:
        # high_value와 low_value를 generate_proposal_with_replaced에 전달
        proposals, scores, pre_proposals, ori_scores = generate_proposal_with_replaced(
            video_feature, query['descriptions'], query['replaced_descriptions'], stride, max_stride, gamma)

        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = proposals[0][:10] * duration
            dynamic_pred = pre_proposals[0][:10] * duration
            scores = scores[0][:10]
            # description별로 정규화. 이렇게 하면 최대 점수를 갖는 구간이 하나의 쿼리에 대해 4개씩(원본 + description 3개) 나옴
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
    
    return answer

