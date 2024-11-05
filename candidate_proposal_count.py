from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer import localize
import os
from llm_prompting import select_proposal, filter_and_integrate
import matplotlib.pyplot as plt
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


# 막대 위에 값 표시하는 함수
def add_value_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

# 히스토그램 그리기 함수
def draw_histograms(data, filename_base):
    # 첫 번째 히스토그램: 1/10 단위로 나눈 것
    plt.figure()
    bars = plt.hist(data, bins=np.arange(0, 1.1, 0.1), edgecolor='black')[2]
    plt.title('Histogram with 1/10 intervals')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    add_value_labels(bars)
    plt.xticks(np.arange(0, 1.1, 0.1))  # x축에 각 구간 표시
    plt.savefig(f'{filename_base}_1_10_intervals.png')
    plt.close()

    # 두 번째 히스토그램: 리스트 값의 최소, 최대값 기준으로 나누기
    min_value, max_value = np.min(data), np.max(data)
    bins = np.linspace(min_value, max_value, 11)
    plt.figure()
    bars = plt.hist(data, bins=bins, edgecolor='black')[2]
    plt.title(f'Histogram with min={min_value:.2f}, max={max_value:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    add_value_labels(bars)
    plt.xticks(bins)  # x축에 각 구간 표시
    plt.savefig(f'{filename_base}_min_max_intervals.png')
    plt.close()


def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    ori_versus_sub_when_proposal_selected_by_ori = []
    ori_versus_sub_when_proposal_selected_by_sub = []

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
            total_length = len(proposals)
            # select_proposal은 각 proposal에 대해서, ((자기 자신과 다른 proposal과의 겹침 정도)^gamma * 다른 proposal의 score)의 합을 계산하여 정렬됨. 따라서, 점수가 높은 proposal과 많이 겹칠수록 높은 점수를 가짐
            proposals = select_proposal(np.array(proposals))

            # 원본 proposal이 선택되었는지 확인
            if proposals[0] in np.array(original_query_proposals):
                ori_versus_sub_when_proposal_selected_by_ori.append(len(original_query_proposals[:7]) / total_length)
            else:
                ori_versus_sub_when_proposal_selected_by_sub.append(len(integrated_sub_query_proposals) / total_length)

    draw_histograms(ori_versus_sub_when_proposal_selected_by_ori, "./candidate_ratio_histogram/activitynet/ori:total candidates ratio-proposal selected from original query")
    draw_histograms(ori_versus_sub_when_proposal_selected_by_sub, "./candidate_ratio_histogram/activitynet/sub:total candidates ratio-proposal selected from sub query")


if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)

    with open(args.llm_output) as f:
        data = json.load(f)
    eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        
