from data_configs import DATASETS
import argparse
import numpy as np
import json
import torch
from tqdm import tqdm
from vlm_localizer_PRVR import generate_segments, video_retrieval, video_retrieval_batch2
import os
import time
import itertools

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--kmeans_gpu', action='store_true', help='Enable use GPU KMeans')
    parser.add_argument('--scene', action='store_true', help='Scene generate')
    parser.add_argument('--output_path', default='default', type=str)

    return parser.parse_args()


def generate_scenes(data, feature_path, output_json_path, hyperparams, kmeans_gpu):
    pbar = tqdm(data.items())
    start_index = 0
    for vid, ann in pbar:
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        scene_segments, proposals, num_frames = generate_segments(video_feature, hyperparams, kmeans_gpu)
        ann['start_index'] = start_index
        ann['scene_segments'] = scene_segments
        ann['proposals'] = proposals
        start_index = start_index + scene_segments[-1][0]

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)



def eval(data, feature_path, stride, hyperparams, kmeans_gpu):
    thresh = [1, 10, 100]
    recall_counts = {k: 0 for k in thresh}
    total_query_num = 0
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        for i in range(len(ann['sentences'])):
            query = ann['sentences'][i]
            sorted_vids = video_retrieval_batch2(data, feature_path, query, stride, hyperparams)

            for k in thresh:
                if vid in sorted_vids[:k]:  # top-k 안에 target_vid가 있는지 확인
                    recall_counts[k] += 1  # 해당 k의 카운트 증가
            
            total_query_num += 1
    
    for th, r in zip(thresh, recall_counts):
        print(f'R@{th}:', r / total_query_num)
        
        
if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    with open(dataset['splits'][args.split]['annotation_file']) as f:
        data = json.load(f)

    if args.scene:
        generate_scenes(data, dataset['feature_path'], args.output_path, dataset['hyper_parameters'], args.kmeans_gpu)
    
    elif args.dataset == 'activitynet':
        with open('dataset/activitynet/test_scene_boundary.json') as f:
            data = json.load(f)
        eval(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.kmeans_gpu)