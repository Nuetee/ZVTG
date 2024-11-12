from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer_global_local_clustering_integrate import localize
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


def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0, dataset_name='charades'):
    candidates_json = {}
    cand_num = 2
    pbar = tqdm(data.items())

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        
        for i in range(len(ann['sentences'])):
            # query
            query_json = [{'descriptions': ann['sentences'][i], 'masked_descriptions': ann['masked'][i], 'gt': ann['timestamps'][i], 'duration': ann['duration']}]
            proposals = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), gamma=0.4, alpha=1, cand_num=cand_num)
            
            gt = ann['timestamps'][i]
            candiates = {
                "sentence_index": i,
                "sentence": ann['sentences'][i],
                "masked_sentence": ann['masked'][i],
                "candidates": proposals.tolist(),
                "duration": duration,
                "timestamps": gt
            }
            if vid not in candidates_json:
                candidates_json[vid] = []
            
            candidates_json[vid].append(candiates)
    
    with open(f"candidates_kmeans-{dataset_name}-cand_num={cand_num}.json", "w") as f:
        json.dump(candidates_json, f, indent=4)

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)

    with open(args.llm_output) as f:
        data = json.load(f)
    eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'], dataset_name=args.dataset)

        
