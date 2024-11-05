from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer import localize
from qvhhighlight_eval import eval_submission
import os
from llm_prompting import select_proposal, filter_and_integrate
import pdb

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


def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    original_query_proposal_ious = [[], [], []]
    sub_query_proposal_ious = [[], [], []]

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
                    # 각 description에 대한 response에서 상위 3개만 proposal에 저장 -> proposals에는 총 9개의 구간 저장
                    for t in range(3):
                        proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
                    proposals = np.array(proposals)
                    # 하나의 sub-query에 대해서 3개의 proposal을 선택
                    sub_query_proposals.append(select_proposal(proposals)[:3])
            else:
                relation = 'single-query'

            # query, 원문 쿼리 하나 + llm 생성 description 3개 => 4개의 description
            query_json = [{'descriptions': ann['sentences'][i]}]
            if 'query_json' in ann['response'][i]:
                query_json += [{'descriptions': q} for q in ann['response'][i]['query_json'][0]['descriptions']]
            answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
            original_query_proposals = []
            for t in range(3):
                original_query_proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
            
            # 총 12개의 proposals에서 앞 7개의 proposals 가져옴 -> 각 description 별 1개씩 + 3개
            integrated_sub_query_proposals = filter_and_integrate(sub_query_proposals, relation)
            proposals = original_query_proposals[:7] + integrated_sub_query_proposals

            # select_proposal은 각 proposal에 대해서, ((자기 자신과 다른 proposal과의 겹침 정도)^gamma * 다른 proposal의 score)의 합을 계산하여 정렬됨. 따라서, 점수가 높은 proposal과 많이 겹칠수록 높은 점수를 가짐
            proposals = select_proposal(np.array(proposals))

            
            gt = ann['timestamps'][i]
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
            
            # 원본 proposal이 선택되었는지 확인
            if proposals[0] in np.array(original_query_proposals):
                if relation == 'single-query':
                    original_query_proposal_ious[0].append(max(iou_, 0))
                elif relation == 'sequentially':
                    original_query_proposal_ious[1].append(max(iou_, 0))
                elif relation == 'simultaneously':
                    original_query_proposal_ious[2].append(max(iou_, 0))
            else:
                if relation == 'single-query':
                    sub_query_proposal_ious[0].append(max(iou_, 0))
                elif relation == 'sequentially':
                    sub_query_proposal_ious[1].append(max(iou_, 0))
                elif relation == 'simultaneously':
                    sub_query_proposal_ious[2].append(max(iou_, 0))

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))
    print('Single query')
    print('# of using original query: ', len(original_query_proposal_ious[0]))
    print('mIoU when using original query: ', sum(original_query_proposal_ious[0]) / len(original_query_proposal_ious[0]) if len(original_query_proposal_ious[0]) > 0 else 0)
    print('# of using sub query integration: ', len(sub_query_proposal_ious[0]))
    print('mIoU when using sub query integration: ', sum(sub_query_proposal_ious[0]) / len(sub_query_proposal_ious[0]) if len(sub_query_proposal_ious[0]) > 0 else 0)
    print('\n')
    print('Sequentially')
    print('# of using original query: ', len(original_query_proposal_ious[1]))
    print('mIoU when using original query: ', sum(original_query_proposal_ious[1]) / len(original_query_proposal_ious[1]) if len(original_query_proposal_ious[1]) > 0 else 0)
    print('# of using sub query integration: ', len(sub_query_proposal_ious[1]))
    print('mIoU when using sub query integration: ', sum(sub_query_proposal_ious[1]) / len(sub_query_proposal_ious[1]) if len(sub_query_proposal_ious[1]) > 0 else 0)
    print('\n')
    print('Simultaneously')
    print('# of using original query: ', len(original_query_proposal_ious[2]))
    print('mIoU when using original query: ', sum(original_query_proposal_ious[2]) / len(original_query_proposal_ious[2]) if len(original_query_proposal_ious[2]) > 0 else 0)
    print('# of using sub query integration: ', len(sub_query_proposal_ious[2]))
    print('mIoU when using sub query integration: ', sum(sub_query_proposal_ious[2]) / len(sub_query_proposal_ious[2]) if len(sub_query_proposal_ious[2]) > 0 else 0)
    print('\n')



if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)

    with open(args.llm_output) as f:
        data = json.load(f)
    eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        
