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

def iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

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

# static score를 기준으로 top1을 생성하고 양 옆으로 확장
def generate_top_proposal(video_features, sentences, stride, max_stride, nms_thresh=0.3):
    scores = calc_scores(video_features, sentences)
    proposal = [torch.argmax(scores), torch.argmax(scores) + 1]
    final_proposal = [torch.argmax(scores), torch.argmax(scores) + 1]
    left_end_score = scores[:, 0].unsqueeze(-1)  # 왼쪽 끝 점수
    right_end_score = scores[:, -1].unsqueeze(-1)  # 오른쪽 끝 점수
    total_score_mean = scores.mean(dim=-1, keepdim=True)  # 전체 구간 점수의 평균
    left_padding_value = torch.min(total_score_mean, left_end_score)  # 왼쪽 패딩 값
    right_padding_value = torch.min(total_score_mean, right_end_score)  # 오른쪽 패딩 값
    static_score = 0
    
    kernel_size = stride
    left_trial_count = 0
    right_trial_count = 0
    max_count = scores.size(-1) // stride
    while True:
        left_start = max(0, proposal[0] - (kernel_size * (left_trial_count + 1)))
        right_end = min(scores.size(-1), proposal[1] + (kernel_size * (right_trial_count + 1)))

        outer_left = torch.mean(scores[0, left_start:proposal[0]])
        if left_start == 0:
            outer_left = left_padding_value
        outer_right = torch.mean(scores[0, proposal[1]:right_end])
        if right_end == scores.size(-1):
            outer_right = right_padding_value
        
        inner_mean = torch.mean(scores[0, proposal[0]:proposal[1]])

        if outer_left < outer_right:
            if inner_mean - outer_left > static_score:
                proposal[0] = left_start
                # final_proposal[0] = left_start
                static_score = inner_mean - outer_left
                left_trial_count = 0
            elif left_trial_count < max_count:
                left_trial_count += 1
            elif right_trial_count < max_count:
                right_trial_count +=1
            else:
                break
        else:
            if inner_mean - outer_right > static_score:
                proposal[1] = right_end
                # final_proposal[1] = right_end
                static_score = inner_mean - outer_right
                right_trial_count = 0
            elif right_trial_count < max_count:
                right_trial_count += 1
            elif left_trial_count < max_count:
                left_trial_count +=1
            else:
                break
    
    final_proposal = [element / scores.size(-1) for element in proposal]
    return final_proposal, static_score


def localize(video_feature, duration, query_json, stride, max_stride):
    answer = []
    for desc_idx, query in enumerate(query_json):
        proposals, scores = generate_top_proposal(video_feature, query['descriptions'], stride, max_stride)
        proposals = [element * duration for element in proposals]
        query['response'] = []
        query['response'].append({
            'start': float(proposals[0]),
            'end': float(proposals[1]),
            'confidence': float(scores),
        })
            
        answer.append(query)
    
    return answer

