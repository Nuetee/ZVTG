from data_configs import DATASETS
import argparse
import numpy as np
import json
import torch
from tqdm import tqdm
from vlm_localizer_TAG_VQA import localize
import os


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--kmeans_gpu', action='store_true', help='Enable use GPU KMeans')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use nonly VLM for evaluation.')

    return parser.parse_args()


def eval_without_llm(data, feature_path, stride, hyperparams, kmeans_gpu):
    pbar = tqdm(data.items())

    for vid, ann in pbar:
        ann.pop('response', None)  # 또는 del ann['response'] (키가 반드시 존재할 경우)
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        ann['proposals'] = []

        for i in range(len(ann['sentences'])):
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)
            ann['proposals'].append(proposals)

    with open("llm_outputs_proposals.json", "w") as f:
        json.dump(data, f, indent=4)

def eval_vqa_frame_5(data, feature_path, stride, hyperparams, kmeans_gpu):
    pbar = tqdm(data.items())

    for vid, ann in pbar:
        ann.pop('response', None)  # 또는 del ann['response'] (키가 반드시 존재할 경우)
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        ann['proposals'] = []

        for i in range(len(ann['sentences'])):
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)
            ann['proposals'].append(proposals)

    with open("llm_outputs_proposals.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    
    if args.llm_output and os.path.exists(args.llm_output):
        with open(args.llm_output) as f:
            data = json.load(f)
        eval_without_llm(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.kmeans_gpu)
