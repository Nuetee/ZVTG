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


def eval_with_llm(data, feature_path, pad_sec=0.0):

    best_recall = np.array([0, 0, 0])
    best_miou = 0
    best_stride = 0
    best_max_stride_factor = 0
    strides = [20]
    max_stride_factor_list = [0.5]

    for stride in strides:
        for max_stride_factor in max_stride_factor_list:
            ious = []
            thresh = np.array([0.3, 0.5, 0.7])
            recall = np.array([0, 0, 0])
            pbar = tqdm(data.items())
            for vid, ann in pbar:
                duration = ann['duration']
                video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

                for i in range(len(ann['sentences'])):
                    # sub queries
                    query_json = [{'descriptions': ann['sentences'][i]}]
                    if stride > int(video_feature.shape[0] * max_stride_factor):
                        stride_ = int(video_feature.shape[0] * max_stride_factor)
                        max_stride = int(video_feature.shape[0] * max_stride_factor)
                    else:
                        stride_ = stride
                        max_stride = int(video_feature.shape[0] * max_stride_factor)
                    if int(video_feature.shape[0] * max_stride_factor) == 0:
                        max_stride = 1
                        stride_ = 1
                    answers = localize(video_feature, duration, query_json, stride_, max_stride)
                    proposals = []
                    for t in range(3):
                        proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]

                    proposals = proposals

                    # select_proposal은 각 proposal에 대해서, ((자기 자신과 다른 proposal과의 겹침 정도)^gamma * 다른 proposal의 score)의 합을 계산하여 정렬됨. 따라서, 점수가 높은 proposal과 많이 겹칠수록 높은 점수를 가짐
                    
                    proposals = select_proposal(np.array(proposals))
                    gt = ann['timestamps'][i]
                    
                    iou_ = calc_iou(proposals[:1], gt)[0]
                    ious.append(max(iou_, 0))
                    recall += thresh <= iou_

                pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

            # mIoU 계산
            current_miou = sum(ious) / len(ious) if len(ious) > 0 else 0
            if current_miou > best_miou:
                best_miou = current_miou
                best_recall = recall
                best_stride = stride
                best_max_stride_factor = max_stride_factor
            
            print(f'mIoU - {stride}/{max_stride_factor}: {current_miou}')
            for th, r in zip(thresh, recall):
                print(f'R@{th}:', r / len(ious))
    
    print('Best mIoU:', best_miou)
    for th, r in zip(thresh, best_recall):
        print(f'R@{th}:', r / len(ious))
    print(f'Best stride: {best_stride}')
    print(f'Best max stride factor: {best_max_stride_factor}')

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    
    with open(args.llm_output) as f:
        data = json.load(f)
    eval_with_llm(data, dataset['feature_path'], dataset['splits'][args.split]['pad_sec'])

        
