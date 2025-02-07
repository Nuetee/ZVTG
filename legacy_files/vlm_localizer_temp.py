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


def nms_with_importance(moments, scores, importance_score, pre_mom, pre_score, thresh):
    scores = scores + pre_score * 0.0
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    pre_mom = pre_mom[ranks]
    importance_score = importance_score[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()

    # 높은 static 점수를 갖는 순으로 구간을 정렬하고 앞뒤로 겹치는 구간이 큰 경우 해당 구간의 suppressed(억제)를 True로
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True
    return moments[~suppressed], pre_mom[~suppressed], scores[~suppressed], importance_score[~suppressed]

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


def generate_proposal(video_features, sentences, stride, max_stride, nms_thresh=0.3):
    # video_features: (93, 32, 256) -> 첫번째 차원은 계속 바뀌는걸로 보아, frame인 듯.
    # sentences: 'person turn a light on.'
    # stride: 20
    # max_stride: 46 (activitynet에 대하여 max_stride == scores.size(-1). 즉 프레임 길이)
    
    # 비디오-텍스트 유사도를 계산하고 0.2 이하값 마스킹
    scores = calc_scores(video_features, sentences)
    masks = (scores > 0.2).float()
    scores = scores * masks
    stride = min(stride, scores.size(-1)//2) # stride가 score 길이 절반을 초과하지 않도록 조정
    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)

    # static scores
    flattened_proposals = [] # 제안 구간
    flattened_scores = [] # 제안 구간에 대한 점수
    flattened_prefix = [] # 동적 인덱스
    flattened_prescore = [] # 동적 점수
    
    for kernel_size in range(stride, min(scores.size(-1)+1, max_stride+1), stride): # stride에 따라 다양한 크기의 커널을 사용하여 구간 제안을 생성
        kernel = torch.ones((1, 1, kernel_size)).to('cuda') # [1, 1, 20], kernal_size = 20
        inner_sum = F.conv1d(scores.view(-1, 1, scores.size(-1)), kernel).view(scores.size(0), -1) # 커널 내부의 점수 합 (모든 가능 구간에 대한 점수 합)
        inner_num = F.conv1d(masks.view(-1, 1, masks.size(-1)), kernel).view(masks.size(0), -1) # 커널 내부의 유효 요소(0.2 이상의 유사도를 갖는 요소)의 개수
        outer_sum = (scores * masks).sum(dim=-1, keepdim=True) - inner_sum # 커널 외부 구간의 점수 합 (모든 가능 구간에 대한 점수 합)
        outer_num = masks.sum(dim=-1, keepdim=True) - inner_num # 커널 외부의 유효요소 개수
        static_scores = inner_sum / kernel_size - outer_sum / outer_num # 내부 평균 점수 - 외부 평균 점수 (Static scoring) [1,74]

        proposals = torch.arange(0, static_scores.size(-1)).to('cuda')
        proposals = torch.stack([proposals, proposals + kernel_size], dim=-1) / scores.size(-1) # 0 ~ 1 정규화 값으로 구간 표현 (ex. [[0, 0.1], [0.1, 0.2], ... [0.9, 1]] -> kernal_size = 0.1), [74, 2]
        
        # 동적구간은 정적구간의 앞에만 등장하도록 설정하므로, 사용되지 않는 뒷부분 삭제
        dynamic_idxs_tmp = dynamic_idxs.narrow(-1, 0, static_scores.size(-1)) # [1, 74]
        dynamic_scores_tmp = dynamic_scores.narrow(-1, 0, static_scores.size(-1)) # [1, 74]
        for idx in range(static_scores.size(0)): # static_scores.size(0) = 1
            mask = static_scores[idx] > -1e3
            if idx >= len(flattened_proposals): # for문의 처음 반복
                flattened_proposals.append(proposals[mask])
                flattened_scores.append(static_scores[idx][mask])
                flattened_prefix.append(dynamic_idxs_tmp[idx][mask])
                flattened_prescore.append(dynamic_scores_tmp[idx][mask])
            else: # for문의 나중 반복
                flattened_proposals[idx] = torch.concat([flattened_proposals[idx], proposals[mask]], dim=0)
                flattened_scores[idx] = torch.concat([flattened_scores[idx], static_scores[idx][mask]], dim=0)
                flattened_prefix[idx] = torch.concat([flattened_prefix[idx], dynamic_idxs_tmp[idx][mask]], dim=0)
                flattened_prescore[idx] = torch.concat([flattened_prescore[idx], dynamic_scores_tmp[idx][mask]], dim=0)
        
        # flattened_proposals는 길이 1짜리 리스트. flattened_proposals[0]에는 계속 proposal들이 append. 처음엔 [74, 2] 두번째 반복엔 [128, 2] ...

    # NMS
    filtered_proposals = []
    filtered_scores = []
    filtered_prefix = []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            # 가능한 모든 proposal(static + dynamic)에 대해, nms
            nms_proposals, nms_prefix, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], flattened_prefix[idx], flattened_scores[idx], nms_thresh)
            filtered_proposals.append(nms_proposals)
            filtered_scores.append(nms_scores)
            filtered_prefix.append(nms_prefix)
        else:
            filtered_proposals.append([])
            filtered_scores.append([])
            filtered_prefix.append([])

    return filtered_proposals, filtered_scores, filtered_prefix, scores

def generate_proposal_my(video_features, sentences, stride, max_stride, nms_thresh=0.3, outer_ratio=1):
    # 비디오-텍스트 유사도를 계산하고 0.2 이하값 마스킹
    scores = calc_scores(video_features, sentences)
    masks = (scores > 0.2).float()
    scores = scores * masks
    stride = min(stride, scores.size(-1)//2)  # stride가 score 길이 절반을 초과하지 않도록 조정
    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)

    # static scores
    flattened_proposals = []  # 제안 구간
    flattened_scores = []  # 제안 구간에 대한 점수
    flattened_prefix = []  # 동적 인덱스
    flattened_prescore = []  # 동적 점수
    
    # region
    for kernel_size in range(stride, min(scores.size(-1) + 1, max_stride + 1), stride):
        kernel = torch.ones((1, 1, kernel_size)).to('cuda')  # [1, 1, kernel_size]
        inner_sum = F.conv1d(scores.view(-1, 1, scores.size(-1)), kernel).view(scores.size(0), -1)

        # 전체 구간 점수 평균 계산
        total_score_mean = scores.mean(dim=-1, keepdim=True)  # 전체 구간 점수의 평균

        # 양쪽 끝 점수 추출
        # 양쪽 끝 점수 추출 (2차원 텐서 처리)
        left_end_score = scores[:, 0].unsqueeze(-1)  # 왼쪽 끝 점수
        right_end_score = scores[:, -1].unsqueeze(-1)  # 오른쪽 끝 점수

        # 각 끝 점수와 전체 평균 중 작은 값으로 패딩 값 결정
        left_padding_value = torch.min(total_score_mean, left_end_score)  # 왼쪽 패딩 값
        right_padding_value = torch.min(total_score_mean, right_end_score)  # 오른쪽 패딩 값

        # 패딩 적용하여 양 옆으로 확장된 텐서 생성
        left_padding_value = left_padding_value.repeat(1, kernel_size * outer_ratio)  # [1, kernel_size]로 반복
        right_padding_value = right_padding_value.repeat(1, kernel_size * outer_ratio)  # [1, kernel_size]로 반복

        # 기존 scores 텐서와 함께 좌우 패딩 값 연결
        padded_scores = torch.cat([left_padding_value, scores, right_padding_value], dim=-1)

        # 좌우 외부 구간의 점수 합 계산
        # 각 inner_sum 구간에 대해 좌우 kernel_size만큼 떨어진 구간 설정
        left_outer_sum = []
        right_outer_sum = []

        for i in range(inner_sum.size(-1)):
            # left 구간 인덱스 설정. kernel 기준 왼쪽에 남은 자리가 없을 때만 패딩값 사용
            if i == 0:
                left_start = 0
            else:
                left_start = max(i, kernel_size * outer_ratio)
            left_end = i + kernel_size * outer_ratio

            left_sum = padded_scores[:, left_start:left_end].sum(dim=-1, keepdim=True)

            # right 구간 인덱스 설정. kernel 기준 오른쪽에 남은 자리가 없을 때만 패딩값 사용
            right_start = i + kernel_size * outer_ratio + kernel_size
            if i == inner_sum.size(-1) - 1:
                right_end = inner_sum.size(-1) + kernel_size * outer_ratio
            else:
                right_end = min(inner_sum.size(-1), right_start + kernel_size * outer_ratio)
            right_sum = padded_scores[:, right_start:right_end].sum(dim=-1, keepdim=True)

            left_outer_sum.append(left_sum)
            right_outer_sum.append(right_sum)
        
        left_outer_sum = torch.cat(left_outer_sum, dim=1)
        right_outer_sum = torch.cat(right_outer_sum, dim=1)
        
        left_outer_avg = left_outer_sum / (kernel_size * outer_ratio)
        right_outer_avg = right_outer_sum / (kernel_size * outer_ratio)

        # 좌우 외부 평균의 평균 계산
        outer_avg = (left_outer_avg + right_outer_avg) / 2.0

        # Static scores 계산: 내부 평균에서 외부 평균을 뺀 값
        static_scores = inner_sum / kernel_size - outer_avg
    # endregion

        proposals = torch.arange(0, static_scores.size(-1)).to('cuda')
        proposals = torch.stack([proposals, proposals + kernel_size], dim=-1) / scores.size(-1)  # 정규화된 구간 표현
        
        # 동적 구간 처리
        dynamic_idxs_tmp = dynamic_idxs.narrow(-1, 0, static_scores.size(-1))
        dynamic_scores_tmp = dynamic_scores.narrow(-1, 0, static_scores.size(-1))
        for idx in range(static_scores.size(0)):
            mask = static_scores[idx] > -1e3
            if idx >= len(flattened_proposals):
                flattened_proposals.append(proposals[mask])
                flattened_scores.append(static_scores[idx][mask])
                flattened_prefix.append(dynamic_idxs_tmp[idx][mask])
                flattened_prescore.append(dynamic_scores_tmp[idx][mask])
            else:
                flattened_proposals[idx] = torch.concat([flattened_proposals[idx], proposals[mask]], dim=0)
                flattened_scores[idx] = torch.concat([flattened_scores[idx], static_scores[idx][mask]], dim=0)
                flattened_prefix[idx] = torch.concat([flattened_prefix[idx], dynamic_idxs_tmp[idx][mask]], dim=0)
                flattened_prescore[idx] = torch.concat([flattened_prescore[idx], dynamic_scores_tmp[idx][mask]], dim=0)

    # NMS 처리
    filtered_proposals = []
    filtered_scores = []
    filtered_prefix = []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            nms_proposals, nms_prefix, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], flattened_prefix[idx], flattened_scores[idx], nms_thresh)
            filtered_proposals.append(nms_proposals)
            filtered_scores.append(nms_scores)
            filtered_prefix.append(nms_prefix)
        else:
            filtered_proposals.append([])
            filtered_scores.append([])
            filtered_prefix.append([])

    return filtered_proposals, filtered_scores, filtered_prefix, scores

def generate_proposal_padded(video_features, sentences, stride, max_stride, nms_thresh=0.3):
    scores = calc_scores(video_features, sentences)
    masks = (scores > 0.2).float()
    scores = scores * masks
    stride = min(stride, scores.size(-1) // 2)
    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)

    flattened_proposals, flattened_scores, flattened_prefix, flattened_prescore = [], [], [], []

    overall_mean = (scores * masks).sum() / masks.sum()  # 전체 평균 계산

    for kernel_size in range(stride, min(scores.size(-1) + 1, max_stride + 1), stride):
        kernel = torch.ones((1, 1, kernel_size)).to('cuda')
        
        # padding 값 설정
        left_end_value = scores[:, 0]
        right_end_value = scores[:, -1]
        left_padding_value = torch.max(left_end_value, overall_mean).view(-1, 1)
        right_padding_value = torch.max(right_end_value, overall_mean).view(-1, 1)
        
        # padding 추가 (양쪽 끝에 설정한 padding 값을 추가함)
        padded_scores = torch.cat([left_padding_value, scores, right_padding_value], dim=-1)
        padded_masks = torch.cat([torch.ones_like(left_padding_value), masks, torch.ones_like(right_padding_value)], dim=-1)

        # inner_sum 계산: padding 제외하고 중간 구간만 추출
        inner_sum_padded = F.conv1d(padded_scores.view(-1, 1, padded_scores.size(-1)), kernel)
        inner_sum = inner_sum_padded[:, :, 1:-1].view(scores.size(0), -1)

        inner_num_padded = F.conv1d(padded_masks.view(-1, 1, padded_masks.size(-1)), kernel)
        inner_num = inner_num_padded[:, :, 1:-1].view(masks.size(0), -1)
        
        outer_sum = (padded_scores * padded_masks).sum(dim=-1, keepdim=True) - inner_sum
        outer_num = padded_masks.sum(dim=-1, keepdim=True) - inner_num

        static_scores = inner_sum / kernel_size - outer_sum / outer_num

        proposals = torch.arange(0, static_scores.size(-1)).to('cuda')
        proposals = torch.stack([proposals, proposals + kernel_size], dim=-1) / scores.size(-1)

        dynamic_idxs_tmp = dynamic_idxs.narrow(-1, 0, static_scores.size(-1))
        dynamic_scores_tmp = dynamic_scores.narrow(-1, 0, static_scores.size(-1))
        
        for idx in range(static_scores.size(0)):
            mask = static_scores[idx] > -1e3
            if idx >= len(flattened_proposals):
                flattened_proposals.append(proposals[mask])
                flattened_scores.append(static_scores[idx][mask])
                flattened_prefix.append(dynamic_idxs_tmp[idx][mask])
                flattened_prescore.append(dynamic_scores_tmp[idx][mask])
            else:
                flattened_proposals[idx] = torch.concat([flattened_proposals[idx], proposals[mask]], dim=0)
                flattened_scores[idx] = torch.concat([flattened_scores[idx], static_scores[idx][mask]], dim=0)
                flattened_prefix[idx] = torch.concat([flattened_prefix[idx], dynamic_idxs_tmp[idx][mask]], dim=0)
                flattened_prescore[idx] = torch.concat([flattened_prescore[idx], dynamic_scores_tmp[idx][mask]], dim=0)

    filtered_proposals, filtered_scores, filtered_prefix = [], [], []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            nms_proposals, nms_prefix, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], flattened_prefix[idx], flattened_scores[idx], nms_thresh)
            filtered_proposals.append(nms_proposals)
            filtered_scores.append(nms_scores)
            filtered_prefix.append(nms_prefix)
        else:
            filtered_proposals.append([])
            filtered_scores.append([])
            filtered_prefix.append([])

    return filtered_proposals, filtered_scores, filtered_prefix, scores

def generate_proposal_modified_static_score(video_features, sentences, stride, max_stride, nms_thresh=0.3):
    scores = calc_scores(video_features, sentences)
    masks = (scores > 0.2).float()
    scores = scores * masks
    stride = min(stride, scores.size(-1) // 2)
    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)

    flattened_proposals, flattened_scores, flattened_prefix, flattened_prescore = [], [], [], []

    for kernel_size in range(stride, min(scores.size(-1) + 1, max_stride + 1), stride):
        kernel = torch.ones((1, 1, kernel_size)).to('cuda')
        inner_sum = F.conv1d(scores.view(-1, 1, scores.size(-1)), kernel).view(scores.size(0), -1)
        inner_num = F.conv1d(masks.view(-1, 1, masks.size(-1)), kernel).view(masks.size(0), -1)
        
        outer_sum = (scores * masks).sum(dim=-1, keepdim=True) - inner_sum
        outer_num = masks.sum(dim=-1, keepdim=True) - inner_num

        left_out = scores[:, :kernel_size] * masks[:, :kernel_size]
        right_out = scores[:, -kernel_size:] * masks[:, -kernel_size:]
        
        # 전체 길이를 기준으로 스케일링 가중치 계산
        total_length = scores.size(-1)
        left_weights = 1 - torch.arange(left_out.size(1)).to('cuda') / total_length
        right_weights = 1 - torch.arange(right_out.size(1)).to('cuda') / total_length

        scaled_left = (left_out * left_weights).sum(dim=1) / left_out.size(1)
        scaled_right = (right_out * right_weights).sum(dim=1) / right_out.size(1)

        outer_mean = (scaled_left + scaled_right) / 2
        static_scores = inner_sum / kernel_size - outer_mean

        proposals = torch.arange(0, static_scores.size(-1)).to('cuda')
        proposals = torch.stack([proposals, proposals + kernel_size], dim=-1) / scores.size(-1)

        dynamic_idxs_tmp = dynamic_idxs.narrow(-1, 0, static_scores.size(-1))
        dynamic_scores_tmp = dynamic_scores.narrow(-1, 0, static_scores.size(-1))
        
        for idx in range(static_scores.size(0)):
            mask = static_scores[idx] > -1e3
            if idx >= len(flattened_proposals):
                flattened_proposals.append(proposals[mask])
                flattened_scores.append(static_scores[idx][mask])
                flattened_prefix.append(dynamic_idxs_tmp[idx][mask])
                flattened_prescore.append(dynamic_scores_tmp[idx][mask])
            else:
                flattened_proposals[idx] = torch.concat([flattened_proposals[idx], proposals[mask]], dim=0)
                flattened_scores[idx] = torch.concat([flattened_scores[idx], static_scores[idx][mask]], dim=0)
                flattened_prefix[idx] = torch.concat([flattened_prefix[idx], dynamic_idxs_tmp[idx][mask]], dim=0)
                flattened_prescore[idx] = torch.concat([flattened_prescore[idx], dynamic_scores_tmp[idx][mask]], dim=0)

    filtered_proposals, filtered_scores, filtered_prefix = [], [], []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            nms_proposals, nms_prefix, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], flattened_prefix[idx], flattened_scores[idx], nms_thresh)
            filtered_proposals.append(nms_proposals)
            filtered_scores.append(nms_scores)
            filtered_prefix.append(nms_prefix)
        else:
            filtered_proposals.append([])
            filtered_scores.append([])
            filtered_prefix.append([])

    return filtered_proposals, filtered_scores, filtered_prefix, scores

def generate_proposal_with_replaced(video_features, sentences, replaced_sentences, stride, max_stride, nms_thresh=0.3):
    # video_features: (93, 32, 256) -> 첫번째 차원은 계속 바뀌는걸로 보아, frame인 듯.
    # sentences: 'person turn a light on.'
    # stride: 20
    # max_stride: 46 (activitynet에 대하여 max_stride == scores.size(-1). 즉 프레임 길이)
    
    # 비디오-텍스트 유사도를 계산하고 0.2 이하값 마스킹
    scores, indices = calc_scores2(video_features, sentences)
    masks = (scores > 0.2).float()
    scores = scores * masks
    
    # toy - algorithm common
    replaced_query_scores = []
    for replaced_element, replaced_queries in replaced_sentences.items():
        if "prepositional" in replaced_element or "subject" in replaced_element or "object" in replaced_element:
            continue
        for replaced_query in replaced_queries:
            if len(replaced_query) == 0:
                continue
            replaced_query_score = calc_scores_with_indices(video_features, [replaced_query], indices)
            replaced_query_scores.append(replaced_query_score)
    # toy - algorithm common

    # toy - alogorithm 0
    # region
    # importance_scores_list = []
    # for replaced_query_score in replaced_query_scores:
    #     # 텐서 연산으로 계산
    #     importance_scores = 1 + (scores - replaced_query_score) * 5
    #     importance_scores_list.append(importance_scores)
    # if len(replaced_query_scores) == 0:
    #     importance_scores = torch.ones_like(scores)
    # else:
    #     importance_scores_tensor = torch.stack(importance_scores_list, dim=0)
    #     importance_scores = importance_scores_tensor.mean(dim=0) # 2D 텐서로 변환
    # endregion
    # toy - alogorithm 0
    
    # toy - algorithm 1, 2
    # region
    # importance_scores_list = []
    # for replaced_query_score in replaced_query_scores:
    #     # 텐서 연산으로 계산
    #     importance_scores = scores - replaced_query_score
    #     importance_scores_list.append(importance_scores)
    # if len(replaced_query_scores) == 0:
    #     importance_scores = torch.ones_like(scores)
    # else:
    #     importance_scores_tensor = torch.stack(importance_scores_list, dim=0)
    #     importance_scores = importance_scores_tensor.mean(dim=0) # 2D 텐서로 변환
    # endregion
    # toy - algorithm 1, 2

    # toy - algorithm 3
    # region
    # importance_scores_list = []
    # for replaced_query_score in replaced_query_scores:
    #     # 텐서 연산으로 계산
    #     importance_scores = scores - replaced_query_score
    #     importance_scores_list.append(importance_scores)
    # endregion
    # toy - algorithm 3

    # toy - algorithm 4
    # region
    importance_scores_list = []
    for replaced_query_score in replaced_query_scores:
        importance_scores = 1 - replaced_query_score / scores
        importance_scores_list.append(importance_scores)
    if len(replaced_query_scores) == 0:
        importance_scores = torch.ones_like(scores)
    # algorithm 4
    else:
        importance_scores_tensor = torch.stack(importance_scores_list, dim=0)
        # importance_scores =  torch.amax(importance_scores_tensor, dim=0)
        importance_scores = importance_scores_tensor.mean(dim=0) # 2D 텐서로 변환
    # algorithm 4
    # endregion
    # toy - algorithm 4

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

        # toy - algorithm 0
        # region
        # importance_inner_sum = F.conv1d(importance_scores.view(-1, 1, importance_scores.size(-1)), kernel).view(importance_scores.size(0), -1)
        # importance_mean = importance_inner_sum / kernel_size
        # static_scores *= (2 * importance_mean)
        # endregion
        # toy - algorithm 0
        
        # toy - algorithm 1
        # region
        # importance_inner_sum = F.conv1d(importance_scores.view(-1, 1, importance_scores.size(-1)), kernel).view(importance_scores.size(0), -1)
        # importance_outer_sum = importance_scores.sum(dim=-1, keepdim=True) - importance_inner_sum
        # importance_outer_num = importance_scores.size(-1) - kernel_size
        # importance_static_scores = importance_inner_sum / kernel_size - importance_outer_sum / importance_outer_num
         # static_scores += importance_static_scores
        # endregion
        # tou - algorithm 1

        # toy - algorithm 2
        # region
        # importance_inner_sum = F.conv1d(importance_scores.view(-1, 1, importance_scores.size(-1)), kernel).view(importance_scores.size(0), -1)
        # importance_outer_sum = importance_scores.sum(dim=-1, keepdim=True) - importance_inner_sum
        # importance_outer_num = importance_scores.size(-1) - kernel_size
        # importance_static_scores = importance_inner_sum / kernel_size - importance_outer_sum / importance_outer_num
        # # toy - algorithm 2-1
        # # normalized_importance_static_scores = (importance_static_scores - importance_static_scores.min()) / (importance_static_scores.max() - importance_static_scores.min())
        # # static_scores *= normalized_importance_static_scores
        # # toy - algorithm 2-1
        # static_scores *= (1 + (2 * importance_static_scores))
        # endregion
        # toy - algorithm 2

        # toy - algorithm 3
        # region
        # importance_static_scores_list = []  # 초기화
        # for importance_scores in importance_scores_list:
        #     importance_inner_sum = F.conv1d(importance_scores.view(-1, 1, importance_scores.size(-1)), kernel).view(importance_scores.size(0), -1)
        #     importance_outer_sum = importance_scores.sum(dim=-1, keepdim=True) - importance_inner_sum
        #     importance_outer_num = importance_scores.size(-1) - kernel_size
        #     importance_static_scores = importance_inner_sum / kernel_size - importance_outer_sum / importance_outer_num
        #     importance_static_scores_list.append(importance_static_scores)
        
        # if len(importance_scores_list) == 0:
        #     importance_static_scores_mean = torch.zeros_like(static_scores)
        # else:
        #     importance_static_scores_tensor = torch.stack(importance_static_scores_list, dim=0)
        #     importance_static_scores_mean = importance_static_scores_tensor.mean(dim=0) # 2D 텐서로 변환
        # normalized_importance_static_scores = (importance_static_scores_mean - importance_static_scores_mean.min()) / (importance_static_scores_mean.max() - importance_static_scores_mean.min())
        
            # toy - algorithm 3
        # static_scores += normalized_importance_static_scores
            # toy - algorithm 3-1
        # static_scores *= (1 + normalized_importance_static_scores)
        # endregion
        # toy - algorithm 3

        # toy - algorithm 4
        # region
        importance_inner_sum = F.conv1d(importance_scores.view(-1, 1, importance_scores.size(-1)), kernel).view(importance_scores.size(0), -1)
        importance_outer_sum = importance_scores.sum(dim=-1, keepdim=True) - importance_inner_sum
        importance_outer_num = importance_scores.size(-1) - kernel_size
        importance_static_scores = importance_inner_sum / kernel_size - importance_outer_sum / importance_outer_num
        # region algorithm 4-1
        # importance_static_scores_list = []  # 초기화
        # for importance_scores in importance_scores_list:
        #     importance_inner_sum = F.conv1d(importance_scores.view(-1, 1, importance_scores.size(-1)), kernel).view(importance_scores.size(0), -1)
        #     importance_outer_sum = importance_scores.sum(dim=-1, keepdim=True) - importance_inner_sum
        #     importance_outer_num = importance_scores.size(-1) - kernel_size
        #     importance_static_scores_el = importance_inner_sum / kernel_size - importance_outer_sum / importance_outer_num
        #     importance_static_scores_list.append(importance_static_scores_el)
        
        # if len(importance_scores_list) == 0:
        #     importance_static_scores = torch.zeros_like(static_scores)
        # else:
        #     importance_static_scores_tensor = torch.stack(importance_static_scores_list, dim=0)
        #     importance_static_scores =  torch.amax(importance_static_scores_tensor, dim=0)
        # static_scores *= (1 + (10 * importance_static_scores))
        # endregion algorithm 4-1
        # endregion
        # toy - algorithm 4

        proposals = torch.arange(0, static_scores.size(-1)).to('cuda')
        proposals = torch.stack([proposals, proposals + kernel_size], dim=-1) / scores.size(-1) # 0 ~ 1 정규화 값으로 구간 표현 (ex. [[0, 0.1], [0.1, 0.2], ... [0.9, 1]] -> kernal_size = 0.1), [74, 2]
        
        # 동적구간은 정적구간의 앞에만 등장하도록 설정하므로, 사용되지 않는 뒷부분 삭제
        # dynamic_idxs_tmp = dynamic_idxs.narrow(-1, 0, static_scores.size(-1)) # [1, 74]
        # dynamic_scores_tmp = dynamic_scores.narrow(-1, 0, static_scores.size(-1)) # [1, 74]
        # for idx in range(static_scores.size(0)): # static_scores.size(0) = 1
        #     mask = static_scores[idx] > -1e3
        #     if idx >= len(flattened_proposals): # for문의 처음 반복
        #         flattened_proposals.append(proposals[mask])
        #         flattened_scores.append(static_scores[idx][mask])
        #         flattened_prefix.append(dynamic_idxs_tmp[idx][mask])
        #         flattened_prescore.append(dynamic_scores_tmp[idx][mask])
        #     else: # for문의 나중 반복
        #         flattened_proposals[idx] = torch.concat([flattened_proposals[idx], proposals[mask]], dim=0)
        #         flattened_scores[idx] = torch.concat([flattened_scores[idx], static_scores[idx][mask]], dim=0)
        #         flattened_prefix[idx] = torch.concat([flattened_prefix[idx], dynamic_idxs_tmp[idx][mask]], dim=0)
        #         flattened_prescore[idx] = torch.concat([flattened_prescore[idx], dynamic_scores_tmp[idx][mask]], dim=0)
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

    flattened_scores = [scaled_flattened_scores[0] * (scaled_flattened_importance_scores[0]**0.2)]

    # NMS
    filtered_proposals = []
    filtered_scores = []
    filtered_importance_scores = []
    filtered_prefix = []
    for idx in range(len(flattened_proposals)):
        if len(flattened_proposals[idx]) > 0:
            # 가능한 모든 proposal(static + dynamic)에 대해, nms
            nms_proposals, nms_prefix, nms_scores, nms_importance_scores = nms_with_importance(flattened_proposals[idx], flattened_scores[idx], flattened_importance_scores[idx], flattened_prefix[idx], flattened_prescore[idx], nms_thresh)
            filtered_proposals.append(nms_proposals)
            filtered_scores.append(nms_scores)
            filtered_importance_scores.append(nms_importance_scores)
            filtered_prefix.append(nms_prefix)
        else:
            filtered_proposals.append([])
            filtered_scores.append([])
            filtered_importance_scores.append([])
            filtered_prefix.append([])
    # NMS
    # filtered_proposals = []
    # filtered_scores = []
    # filtered_prefix = []
    # for idx in range(len(flattened_proposals)):
    #     if len(flattened_proposals[idx]) > 0:
    #         # 가능한 모든 proposal(static + dynamic)에 대해, nms
    #         nms_proposals, nms_prefix, nms_scores = nms(flattened_proposals[idx], flattened_scores[idx], flattened_prefix[idx], flattened_scores[idx], nms_thresh)
    #         filtered_proposals.append(nms_proposals)
    #         filtered_scores.append(nms_scores)
    #         filtered_prefix.append(nms_prefix)
    #     else:
    #         filtered_proposals.append([])
    #         filtered_scores.append([])
    #         filtered_prefix.append([])

    return filtered_proposals, filtered_scores, filtered_prefix, scores

def localize(video_feature, duration, query_json, stride, max_stride):
    answer = []
    for query in query_json:
        proposals, scores, pre_proposals, ori_scores = generate_proposal(video_feature, query['descriptions'], stride, max_stride)
        # proposals, scores, pre_proposals, ori_scores = generate_proposal_my(video_feature, query['descriptions'], stride, max_stride, outer_ratio=2)
        try:
            if len(proposals[0]) == 0:
                static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
                dynamic_pred = np.array([0.0, 0.0, 0.0])
                scores = np.array([1.0, 1.0, 1.0])
            else:
                static_pred = proposals[0][:10] * duration
                dynamic_pred = pre_proposals[0][:10] * duration
                scores = scores[0][:10]
                # description별로 정규화. 이렇게하면 최대 점수를 갖는 구간이 하나의 쿼리에 대해 4개씩(원본 + description 3개) 나옴
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
        except:
            import pdb;pdb.set_trace()
    
    return answer

def localize_mod(video_feature, duration, query_json,stride, max_stride):
    answer = []
    for query in query_json:
        proposals, scores, pre_proposals, ori_scores = generate_proposal(video_feature, query['descriptions'], query['replaced_descriptions'], stride, max_stride)
        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = proposals[0][:10] * duration
            dynamic_pred = pre_proposals[0][:10] * duration
            scores = scores[0][:10]
            # description별로 정규화. 이렇게하면 최대 점수를 갖는 구간이 하나의 쿼리에 대해 4개씩(원본 + description 3개) 나옴
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

def localize_from_mask(video_feature, duration, query_json,stride, max_stride):
    answer = []
    for query in query_json:
        proposals, scores, pre_proposals, ori_scores = generate_proposal(video_feature, query['descriptions'], stride, max_stride)
        verb_masked_descriptions = query['replaced_descriptions']['verb_masked']
        all_verb_proposals = []
        all_verb_dynamics = []
        all_verb_scores = []

        proposal_top_num = 3
        verb_proposal_top_num = 6
        for verb_masked_description in verb_masked_descriptions:
            verb_proposals, verb_scores, verb_pre_proposals, ori_scores = generate_proposal(video_feature, verb_masked_description, stride, max_stride)
            if len(verb_proposals[0]) != 0:
                verb_scores = verb_scores[0][:verb_proposal_top_num]
                verb_scores = verb_scores / verb_scores.max()

                all_verb_dynamics.extend(verb_pre_proposals[0][:verb_proposal_top_num])
                all_verb_proposals.extend(verb_proposals[0][:verb_proposal_top_num])
                all_verb_scores.extend(verb_scores)

        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            # Normalize scores and verb_scores separately
            scores = scores[0][:proposal_top_num]
            scores = scores / scores.max()
            
            # Select top 10 proposals and verb proposals
            if len(all_verb_proposals) > 0:
                all_verb_proposals = torch.stack(all_verb_proposals)
                all_verb_dynamics = torch.stack(all_verb_dynamics)
                all_verb_scores = torch.stack(all_verb_scores)

                static_pred = torch.cat((proposals[0][:proposal_top_num], all_verb_proposals)) * duration
                dynamic_pred = torch.cat((pre_proposals[0][:proposal_top_num], all_verb_dynamics)) * duration

                # Concatenate normalized scores
                scores = torch.cat((scores, all_verb_scores))
            else:
                static_pred = proposals[0][:proposal_top_num] * duration
                dynamic_pred = pre_proposals[0][:proposal_top_num] * duration

        query['response'] = []
        for i in range(len(static_pred)):
            query['response'].append({
                'start': float(static_pred[i][0]),
                'static_start': float(static_pred[i][0]),
                'end': float(static_pred[i][1]),
                'confidence': float(scores[i])
            })
        answer.append(query)
    
    return answer