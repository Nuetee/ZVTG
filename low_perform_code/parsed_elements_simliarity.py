from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import torch
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from vlm_localizer import localize
import pdb

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda', is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])

def select_proposal(inputs, gamma=0.6):
    weights = inputs[:, -1].clip(min=0)
    proposals = inputs[:, :-1]
    scores = np.zeros_like(weights)

    for j in range(scores.shape[0]):
        iou = calc_iou(proposals, proposals[j])
        scores[j] += (iou ** gamma * weights).sum()

    idx = np.argsort(-scores)
    return inputs[idx], idx

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use only VLM for evaluation.')

    return parser.parse_args()

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

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

# 유사도 저장용 딕셔너리
all_similarities = {}

def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            # query, 원문 쿼리 하나 + llm 생성 description 3개 => 4개의 description
            if 'query_json' in ann['response'][i]:
                relation = ann['response'][i]['relationship']
                if relation == 'simultaneously' or relation == 'sequentially':
                    continue

                query_json = [{'descriptions': ann['sentences'][i]}]
                parsed_queries = [ann['parsed_query'][i] if ann['parsed_query'][i] else None]
                if 'query_json' in ann['response'][i]:
                    query_json += [{'descriptions': q} for q in ann['response'][i]['query_json'][0]['descriptions']]
                    parsed_queries += [p for p in ann['response'][i]['query_json'][0]['parsed_query']]
                answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
                proposals = []
                proposal_to_description_map = []  # description 인덱스를 추적하기 위한 리스트
                
                for t in range(3):
                    for idx, p in enumerate(answers):
                        if len(p['response']) > t:
                            proposals.append([p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']])
                            proposal_to_description_map.append((idx, query_json[idx]['descriptions']))  # 해당 proposal의 description 저장

                # 총 12개의 proposals에서 앞 7개의 proposals 가져옴 -> 각 description 별 1개씩 + 3개
                proposals = proposals[:7]
                proposal_to_description_map = proposal_to_description_map[:7]

                proposals, selected_idx = select_proposal(np.array(proposals))

                # 가장 높은 score로 선택된 proposal에 해당하는 description을 query_json에서 찾음
                selected_description_idx, selected_description = proposal_to_description_map[selected_idx[0]]

                gt = ann['timestamps'][i]
                iou_ = calc_iou(proposals[:1], gt)[0]
                ious.append(max(iou_, 0))
                recall += thresh <= iou_

                parsed_query = parsed_queries[selected_description_idx]
                
                # parsed_query에서 subject, verb, object, location 추출
                subject = parsed_query.get("subject", "None")
                verb = parsed_query.get("verb", "None")
                obj = parsed_query.get("object", "None")
                prepositional_phrase = parsed_query.get("prepositional phrase", "None")

                # 각 요소의 텍스트를 개별적으로 구성
                elements = {
                    "sentence": selected_description,
                    "subject": subject,
                    "verb": verb,
                    "object": obj,
                    "prepositional phrase": prepositional_phrase
                }

                # 각 요소별로 유사도 계산
                scores = {}
                for element_name, element_text in elements.items():
                    if not isinstance(element_text, str) or element_text == "None":
                        # print(f"Skipping element_text 'None' for {element_name} in video {vid}, sentence {i}")
                        continue
                    
                    # 요소 텍스트와 video_feature 간의 유사도 계산
                    element_score = calc_scores(video_feature, [element_text])
                    scores[element_name] = element_score.squeeze().cpu().numpy().tolist()  # 1차원 리스트로 변환

                # 문장 및 관련 메타데이터와 함께 유사도를 저장
                similarity_entry = {
                    "sentence_index": i,
                    "sentence": selected_description,
                    "timestamps": ann.get("timestamps", [])[i],
                    "proposal": proposals[:1].tolist(),
                    "duration": duration,
                    "similarity_scores": scores,
                    "elements": elements
                }
                
                # vid가 이미 all_similarities에 있는지 확인하고, 없으면 초기화
                if vid not in all_similarities:
                    all_similarities[vid] = []

                # 유사도 정보를 리스트에 추가
                all_similarities[vid].append(similarity_entry)
    
    # 유사도 파일로 저장
    with open("similarities_per_frame.json", "w") as f:
        json.dump(all_similarities, f, indent=4)

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    
    with open(args.llm_output) as f:
        data = json.load(f)
    eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
