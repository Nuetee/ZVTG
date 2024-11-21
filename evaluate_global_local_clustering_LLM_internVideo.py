from data_configs import DATASETS
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer_global_local_clustering_internVideo import localize
from qvhhighlight_eval import eval_submission
import os
from llm_prompting import select_proposal, filter_and_integrate

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

def eval_with_llm(data, feature_path, stride, max_stride_factor):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    pbar = tqdm(data.items())
    gamma=0.2
    cand_num=12
    kmeans_k=9
    prior=0.5
    temporal_window_size=21
    use_llm=True

    
    for vid, ann in pbar:
        duration = ann['duration']

        file_path = os.path.join(feature_path, vid + '.pt')
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            # print(f"File not found: {file_path}")
            continue  # 다음 반복으로 넘어감
        # 파일이 존재하면 로드
        video_feature = torch.load(file_path)

        # video_feature = torch.load(os.path.join(feature_path, vid+'.pt'))
        if isinstance(video_feature, torch.Tensor):
            video_feature = video_feature.numpy()  # Convert PyTorch tensor to NumPy array
        else:
            video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            # query
            query_json = [{'descriptions': ann['sentences'][i], 'gt': ann['timestamps'][i], 'duration': ann['duration']}]
            proposals = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), gamma=gamma, cand_num=cand_num, kmeans_k=kmeans_k, prior=prior, temporal_window_size=temporal_window_size, use_llm=use_llm)
            
            if 'query_json' in ann['response'][i]:
                for j in range(len(ann['response'][i]['query_json'][0]['descriptions'])):
                    query_json = [{'descriptions': ann['response'][i]['query_json'][0]['descriptions'][j], 'gt': ann['timestamps'][i], 'duration': ann['duration']}]
                    proposals += localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), gamma=gamma, cand_num=cand_num, kmeans_k=kmeans_k, prior=prior, temporal_window_size=temporal_window_size, use_llm=use_llm)

            proposals = select_proposal(np.array(proposals))
            gt = ann['timestamps'][i]
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

    eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'])
        
