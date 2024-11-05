from data_configs import DATASETS
import argparse
import matplotlib.pyplot as plt
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

import matplotlib.pyplot as plt
import numpy as np

def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    iou_list = []  # (vid, sentence, IoU) 값을 저장할 리스트

    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

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
                for j in range(1, len(ann['response'][i]['query_json'])):
                    query_json = [{'descriptions': q} for q in ann['response'][i]['query_json'][j]['descriptions']]
                    answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
                    proposals = []
                    for t in range(3):
                        proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
                    proposals = np.array(proposals)
                    sub_query_proposals.append(select_proposal(proposals)[:3])
            else:
                relation = 'single-query'

            query_json = [{'descriptions': ann['sentences'][i]}]
            if 'query_json' in ann['response'][i]:
                query_json += [{'descriptions': q} for q in ann['response'][i]['query_json'][0]['descriptions']]
            answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
            proposals = []
            for t in range(3):
                proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]

            original_query_proposals = proposals[:7]
            integrated_sub_query_proposals = filter_and_integrate(sub_query_proposals, relation)
            proposals_2 = original_query_proposals + integrated_sub_query_proposals
            proposals_2 = np.array(proposals_2)

            proposals = proposals[:7] + filter_and_integrate(sub_query_proposals, relation)
            proposals = select_proposal(np.array(proposals))

            gt = ann['timestamps'][i]
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

            # IoU 값, vid, sentence를 리스트에 저장
            sentence = ann['sentences'][i]
            iou_list.append((vid, sentence, iou_))

            gt_lengths.append(gt[1] - gt[0])

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

    # IoU 리스트를 JSON 파일로 저장
    with open('iou_list.json', 'w') as f:
        json.dump(iou_list, f, indent=4)

    # IoU 리스트의 최대값 기준으로 10등분한 히스토그램 그리기
    max_iou = max([iou for _, _, iou in iou_list])  # IoU 값만 추출하여 최대값 계산
    bins = np.linspace(0, max_iou, 11)  # 0부터 max_iou까지 10등분

    # IoU 값만 추출하여 히스토그램 그리기
    iou_values = [iou for _, _, iou in iou_list]
    mIoU = np.mean(iou_values)
    total_samples = len(iou_values)

    plt.hist(iou_values, bins=bins, edgecolor='black')
    plt.title(f'IoU Distribution (Total Samples: {total_samples})')
    plt.xlabel('IoU')
    plt.ylabel('Number of Samples')

    # mIoU를 점선으로 표시
    plt.axvline(mIoU, color='r', linestyle='--', label=f'mIoU = {mIoU:.2f}')

    # 히스토그램을 이미지로 저장
    plt.savefig('iou_histogram.png')  # 이미지 파일명 지정
    plt.close()  # 메모리 관리를 위해 plt를 닫습니다.

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    if args.llm_output and os.path.exists(args.llm_output):
        with open(args.llm_output) as f:
            data = json.load(f)
        eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
    else:
        with open(dataset['splits'][args.split]['annotation_file']) as f:
            data = json.load(f)
        eval(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
