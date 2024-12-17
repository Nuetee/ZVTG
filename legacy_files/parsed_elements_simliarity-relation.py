from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from vlm_localizer import localize, calc_scores
from llm_prompting import filter_and_integrate2
import pdb

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
                if relation != 'simultaneously' and relation != 'sequentially':
                    continue
                
                # sub-query proposal
                sub_query_proposals = []
                sub_query_descriptions = []
                for j in range(1, len(ann['response'][i]['query_json'])):
                    query_json = [{'descriptions': q} for q in ann['response'][i]['query_json'][j]['descriptions']]
                    answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
                    proposals = []
                    # 각 description에 대한 response에서 상위 3개만 proposal에 저장 -> proposals에는 총 9개의 구간 저장
                    proposal_to_description_map = []  # description 인덱스를 추적하기 위한 리스트
                    
                    for t in range(3):
                        for idx, p in enumerate(answers):
                            if len(p['response']) > t:
                                proposals.append([p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']])
                                proposal_to_description_map.append((idx, query_json[idx]['descriptions']))

                    proposals, selected_idx = select_proposal(np.array(proposals))
                    selected_description = [proposal_to_description_map[i] for i in selected_idx]
                    # 하나의 sub-query에 대해서 3개의 proposal을 선택
                    sub_query_proposals.append(proposals[:3])
                    sub_query_descriptions.append(selected_description[:3])

                integrated_sub_query_proposals, selected_indices = filter_and_integrate2(sub_query_proposals, relation)

                final_sub_query_descriptions = []
                for selected_idx in selected_indices:
                    description_combination = []
                    for sub_query_proposal in selected_idx:
                        query_loc = sub_query_proposal[0]
                        desc_loc = sub_query_proposal[1]
                        description_combination.append(sub_query_descriptions[query_loc][desc_loc])
                    final_sub_query_descriptions.append(description_combination)
                

                if len(integrated_sub_query_proposals) != 0:
                    _, selected_idx = select_proposal(np.array(integrated_sub_query_proposals))
                    selected_sub_description = final_sub_query_descriptions[selected_idx[0]]
                    selected_sub_description = [text for _, text in selected_sub_description]
                else:
                    selected_sub_description = []

                # original-query proposal
                query_json = [{'descriptions': ann['sentences'][i]}]
                if 'query_json' in ann['response'][i]:
                    query_json += [{'descriptions': q} for q in ann['response'][i]['query_json'][0]['descriptions']]
                answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
                proposals = []
                original_query_descriptions = []  # description 인덱스를 추적하기 위한 리스트
                
                for t in range(3):
                    for idx, p in enumerate(answers):
                        if len(p['response']) > t:
                            proposals.append([p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']])
                            original_query_descriptions.append((idx, query_json[idx]['descriptions']))  # 해당 proposal의 description 저장

                # 총 12개의 proposals에서 앞 7개의 proposals 가져옴 -> 각 description 별 1개씩 + 3개
                original_query_proposals = proposals[:7]
                original_query_descriptions = original_query_descriptions[:7]
                _, selected_idx = select_proposal(np.array(original_query_proposals))

                # 가장 높은 score로 선택된 proposal에 해당하는 description을 query_json에서 찾음
                selected_description_idx, selected_description = original_query_descriptions[selected_idx[0]]
                
                proposals = original_query_proposals + integrated_sub_query_proposals
                proposals, selected_idx = select_proposal(np.array(proposals))

                is_ori = True
                if selected_idx[0] >= len(original_query_proposals):
                    is_ori = False

                gt = ann['timestamps'][i]
                iou_ = calc_iou(proposals[:1], gt)[0]
                ious.append(max(iou_, 0))
                
                elements = {
                    "sentence": selected_description,
                }
                for idx, sub_description in enumerate(selected_sub_description, start=1):
                    key = f"sub_description_{idx}"
                    elements[key] = sub_description
                                                
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
                    "relationship": relation,
                    "sentence": selected_description,
                    "timestamps": ann.get("timestamps", [])[i],
                    "proposal": proposals[:1].tolist(),
                    "duration": duration,
                    "similarity_scores": scores,
                    "elements": elements,
                    "selected_by_ori": is_ori
                }
                for idx, sub_description in enumerate(selected_sub_description, start=1):
                    key = f"sub_description_{idx}"
                    similarity_entry[key] = sub_description
                                                
                # vid가 이미 all_similarities에 있는지 확인하고, 없으면 초기화
                if vid not in all_similarities:
                    all_similarities[vid] = []

                # 유사도 정보를 리스트에 추가
                all_similarities[vid].append(similarity_entry)
    
    # 유사도 파일로 저장
    with open("similarities_per_frame-all_relation.json", "w") as f:
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
