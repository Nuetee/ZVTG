from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer_Final import localize
import os
from llm_prompting import select_proposal

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--use_llm', action='store_true', help='Enable use llm')
    parser.add_argument('--tckmeans', action='store_true', help='Enable use GPU KMeans')
    parser.add_argument('--duration', action='store_true', help='Video duration analysis')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use nonly VLM for evaluation.')
    parser.add_argument('--api', action='store_true', help='Enable use GPT API call')

    return parser.parse_args()


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def eval_without_llm(data, feature_path, stride, hyperparams, tckmeans):
    QGA_window_size_list = [1, 3, 5, 7, 9, 11, 13 ,15]

    for window_size in QGA_window_size_list:
        print("QGA window size: ", window_size)
        ious = []
        thresh = np.array([0.3, 0.5, 0.7])
        recall = np.array([0, 0, 0])
        pbar = tqdm(data.items())
        hyperparams["temporal_window_size"] = window_size

        for vid, ann in pbar:
            duration = ann['duration']
            video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
            
            for i in range(len(ann['sentences'])):
                gt = ann['timestamps'][i]
                query_json = [{'descriptions': ann['sentences'][i]}]
                proposals = localize(video_feature, duration, query_json, stride, hyperparams, tckmeans)
                proposals = select_proposal(np.array(proposals))

                iou_ = calc_iou(proposals[:1], gt)[0]
                ious.append(max(iou_, 0))
                recall += thresh <= iou_

            pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})


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
    eval_without_llm(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.tckmeans)