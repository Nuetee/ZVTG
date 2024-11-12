from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_generate_top import localize
import os
import torch
import numpy as np
import torch.nn.functional as F
from llm_prompting import select_proposal, filter_and_integrate
from lavis.models import load_model_and_preprocess
from torchvision import transforms
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
def get_similarity(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            if 'query_json' in ann['response'][i]:
                relation = ann['response'][i]['relationship']
                if relation == 'simultaneously' or relation == 'sequentially':
                    continue
                # query, 원문 쿼리 하나 + llm 생성 description 3개 => 4개의 description
                query_json = [{'descriptions': ann['sentences'][i]}]
                parsed_queries = [ann['parsed_query'][i] if ann['parsed_query'][i] else None]
                if 'query_json' in ann['response'][i]:
                    query_json += [{'descriptions': q} for q in ann['response'][i]['query_json'][0]['descriptions']]
                    parsed_queries += [p for p in ann['response'][i]['query_json'][0]['parsed_query']]
                answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
                proposals = []
                proposal_to_description_map = []  # description 인덱스를 추적하기 위한 리스트

                for idx, p in enumerate(answers):
                    proposals.append([p['response'][0]['start'], p['response'][0]['end'], p['response'][0]['confidence']])
                    proposal_to_description_map.append((idx, query_json[idx]['descriptions']))
                
                
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
    
        pbar.set_postfix({"mIoU": sum(ious) / len(ious) if len(ious) != 0 else 0})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

    # 유사도 파일로 저장
    with open("similarities_per_frame-top_similarity.json", "w") as f:
        json.dump(all_similarities, f, indent=4)


def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    # region
    single_query_ious = []
    simultaneously_ious = []
    sequentially_ious = []
    single_query_ious_relation = []
    simultaneously_ious_relation = []
    sequentially_ious_relation = []
    single_query_recall = np.array([0, 0, 0])
    simultaneously_recall = np.array([0, 0, 0])
    sequentially_recall = np.array([0, 0, 0])
    # k=0
    # endregion
    ious = []
    ious_relation = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    recall_relation = np.array([0, 0, 0])
    
    gt_lengths = []

    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            # sub queries
            sub_query_proposals = []
            if 'query_json' in ann['response'][i]:
                relation = ann['response'][i]['relationship']
                # j의 range가 1부터 시작하는 이유는 0번째는 sub-query가 아닌 전체 query이기 때문
                for j in range(1, len(ann['response'][i]['query_json'])):
                    query_json = [{'descriptions': q} for q in ann['response'][i]['query_json'][j]['descriptions']]
                    # 하나의 description에 대해 10개 이하의 response(st:end, confidence) / 10개 이하인 이유는 10개를 뽑지만 nms에 의해 억제된 경우 그 이하의 proposal들이 반환되기 때문
                    answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
                    proposals = []
                    proposals = [[p['response'][0]['start'], p['response'][0]['end'], p['response'][0]['confidence']] for p in answers]
                    proposals = np.array(proposals)
                    sub_query_proposals.append(proposals)
                    
            else:
                relation = 'single-query'

            # query, 원문 쿼리 하나 + llm 생성 description 3개 => 4개의 description
            query_json = [{'descriptions': ann['sentences'][i]}]
            if 'query_json' in ann['response'][i]:
                query_json += [{'descriptions': q} for q in ann['response'][i]['query_json'][0]['descriptions']]
            answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
            proposals = []
            proposals = [[p['response'][0]['start'], p['response'][0]['end'], p['response'][0]['confidence']] for p in answers]
            
            # region
            original_query_proposals = proposals[:7]
            integrated_sub_query_proposals = filter_and_integrate(sub_query_proposals, relation)
            proposals_2 = original_query_proposals + integrated_sub_query_proposals
            proposals_2 = np.array(proposals_2)
            # endregion
            
            # 총 12개의 proposals에서 앞 7개의 proposals 가져옴 -> 각 description 별 1개씩 + 3개
            proposals = proposals[:7] + filter_and_integrate(sub_query_proposals, relation)

            # select_proposal은 각 proposal에 대해서, ((자기 자신과 다른 proposal과의 겹침 정도)^gamma * 다른 proposal의 score)의 합을 계산하여 정렬됨. 따라서, 점수가 높은 proposal과 많이 겹칠수록 높은 점수를 가짐
            proposals, _ = select_proposal(np.array(proposals))
            gt = ann['timestamps'][i]
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
            
            integrated_sub_query_proposals = np.array(integrated_sub_query_proposals)
            if len(integrated_sub_query_proposals) != 0:
                iou_relation = calc_iou(integrated_sub_query_proposals[:1], gt)[0]
            else:
                iou_relation = 0
            ious_relation.append(max(iou_relation, 0))
            recall_relation += thresh <= iou_relation

            gt_lengths.append(gt[1] - gt[0])

            # region
            if relation == 'single-query':
                single_query_ious.append(max(iou_, 0))
                single_query_recall += thresh <= iou_

                single_query_ious_relation.append(max(iou_relation, 0))
            elif relation == 'simultaneously':
                simultaneously_ious.append(max(iou_, 0))
                simultaneously_recall += thresh <= iou_

                simultaneously_ious_relation.append(max(iou_relation, 0))
            elif relation == 'sequentially':
                sequentially_ious.append(max(iou_, 0))
                sequentially_recall += thresh <= iou_

                sequentially_ious_relation.append(max(iou_relation, 0))
            # endregion

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), "mIoU - relation":  sum(ious_relation) / len(ious_relation)})


    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

    # region
    print('Single query - mIoU:', sum(single_query_ious) / len(single_query_ious))
    for th, r in zip(thresh, single_query_recall):
        print(f'Single query R@{th}:', r / len(single_query_ious))
    print('Single query data number:', len(single_query_ious))

    print('Simultaneously - mIoU:', sum(simultaneously_ious) / len(simultaneously_ious))
    for th, r in zip(thresh, simultaneously_recall):
        print(f'Simultaneously R@{th}:', r / len(simultaneously_ious))
    print('Simultaneously query data number:', len(simultaneously_ious))

    print('Sequentially - mIoU:', sum(sequentially_ious) / len(sequentially_ious))
    for th, r in zip(thresh, sequentially_recall):
        print(f'Sequentially R@{th}:', r / len(sequentially_ious))
    print('Sequentially query data number:', len(sequentially_ious))
    
    print('Single query relation - mIoU:', sum(single_query_ious_relation) / len(single_query_ious_relation))
    print('Simultaneously relation - mIoU:', sum(simultaneously_ious_relation) / len(simultaneously_ious_relation))
    print('Sequentially relation - mIoU:', sum(sequentially_ious_relation) / len(sequentially_ious_relation))
    # endregion

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    
    with open(args.llm_output) as f:
        data = json.load(f)
    get_similarity(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
