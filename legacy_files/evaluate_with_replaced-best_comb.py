from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer_best_comb import localize2, calc_scores
from qvhhighlight_eval import eval_submission
from llm_prompting import select_proposal, filter_and_integrate
import os
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
    best_miou = 0
    best_gamma = 0.1

    # 그리드 서치를 위한 파라미터 범위
    gamma = np.linspace(0.1, 1, 10)

    # 전체 데이터셋에 대해 최적의 조합을 찾기 위해 반복
    for current_gamma in gamma:
        ious = []
        recall = np.array([0, 0, 0])
        thresh = np.array([0.3, 0.5, 0.7])

        # 데이터셋의 모든 비디오에 대해 평가
        pbar = tqdm(data.items())
        for vid, ann in pbar:
            duration = ann['duration']
            video_feature = np.load(os.path.join(feature_path, vid + '.npy'))

            for i in range(len(ann['revise_and_replace'])):
                query_json = [{'descriptions': ann['revise_and_replace'][i]["revised_query"][0], 'replaced_descriptions': ann['masked'][i]}]

                answers = localize2(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), current_gamma)
                proposals = []
                for t in range(3):
                    proposals += [[p['response'][t]['static_start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]

                proposals = np.array(proposals)
                gt = ann['timestamps'][i]

                iou_ = calc_iou(proposals[:1], gt)[0]
                ious.append(max(iou_, 0))
                recall += thresh <= iou_

        # mIoU 계산
        current_miou = sum(ious) / len(ious) if len(ious) > 0 else 0
        if current_miou > best_miou:
            best_miou = current_miou
            best_gamma = current_gamma
        print(f'mIoU - {current_gamma}: {current_miou}')

    # 최적의 조합 출력
    print('Best mIoU:', best_miou)
    print('Best Gamma:', best_gamma)

    # 최적의 조합에 대해 최종 mIoU 계산 및 출력
    ious = []
    recall = np.array([0, 0, 0])
    thresh = np.array([0.3, 0.5, 0.7])
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid + '.npy'))

        for i in range(len(ann['revise_and_replace'])):
            query_json = [{'descriptions': ann['revise_and_replace'][i]["revised_query"][0], 'replaced_descriptions': ann['masked'][i]}]

            answers = localize2(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), best_combination[0], best_combination[1])
            proposals = []
            for t in range(3):
                proposals += [[p['response'][t]['static_start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]

            proposals = np.array(proposals)
            gt = ann['timestamps'][i]

            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    
    with open(args.llm_output) as f:
        data = json.load(f)
    eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])