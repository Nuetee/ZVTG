from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer import localize_mod, calc_scores
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

# def select_proposal(inputs, gamma=0.6):
#     weights = inputs[:, 2].clip(min=0)
#     proposals = inputs[:, 0:2]
#     importance_scores = inputs[:, 3:]
#     scores = np.zeros_like(weights)

#     import pdb
#     pdb.set_trace()
#     for j in range(scores.shape[0]):
#         iou = calc_iou(proposals, proposals[j])
#         importance_score = np.mean(importance_scores[j])
#         scores[j] += (iou ** gamma * weights).sum()
#         scores[j] *= importance_score

#     idx = np.argsort(-scores)
#     return inputs[idx]

def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    # toy
    max_ious = []
    # toy
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['revise_and_replace'])):
            query_json = [{'descriptions': ann['revise_and_replace'][i]["revised_query"][0], 'replaced_descriptions': ann['masked'][i]}]
        # for i in range(len(ann['sentences'])):
        #     relation = ann['response'][i]['relationship']
        #     if relation != 'simultaneously' and relation != 'sequentially':
        #         continue
        #     query_json = [{'descriptions': ann['revise_and_replace'][i]["revised_query"][0]}]
        #     query_json[0]['replaced_descriptions'] = [ann['response'][i]['query_json'][j]['descriptions'][0] for j in range(1, len(ann['response'][i]['query_json']))]

            answers = localize_mod(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
            proposals = []
            for t in range(3):
            # for t in range(len(answers[0]['response'])):
                proposals += [[p['response'][t]['static_start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
            
            proposals = select_proposal(np.array(proposals))
            # proposals = np.array(proposals)
            
            gt = ann['timestamps'][i]
            # toy
            max_iou = 0
            for i in range(proposals.shape[0]):
                iou = calc_iou(proposals[i:i+1], gt)[0]
                if max_iou < iou:
                    max_iou = iou
            max_ious.append(max_iou)
            # toy
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
        if len(ious) > 0:
            pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    # toy
    print('max_mIoU:', sum(max_ious) / len(max_ious))
    # toy
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