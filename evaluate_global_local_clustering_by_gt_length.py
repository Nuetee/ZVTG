from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer_global_local_clustering_final import localize
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

def calc_iou2(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = e - s
    return inter.clip(min=0) / union

def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    pbar = tqdm(data.items())

   # 구간을 20% 미만, 20~30%, 30~40%, 40% 이상으로 나누어 초기화
    interval_ious = {'<20%': [], '20-30%': [], '30-40%': [], '>=40%': []}
    interval_recalls = {'<20%': np.array([0, 0, 0]), '20-30%': np.array([0, 0, 0]), '30-40%': np.array([0, 0, 0]), '>=40%': np.array([0, 0, 0])}
    interval_counts = {'<20%': 0, '20-30%': 0, '30-40%': 0, '>=40%': 0}  # 각 구간의 개수를 기록

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        num_frames = video_feature.shape[0]
        for i in range(len(ann['sentences'])):
            # query
            query_json = [{'descriptions': ann['sentences'][i], 'gt': ann['timestamps'][i], 'duration': ann['duration']}]
            proposals = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), gamma=0.2, cand_num=12, kmeans_k=7)
    
            gt = ann['timestamps'][i]
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

            # gt의 상대적 길이에 따른 비율 계산 및 구간별 IoU와 recall 기록
            relative_length = (gt[1] - gt[0]) / duration
            if relative_length < 0.2:
                interval_key = '<20%'
            elif 0.2 <= relative_length < 0.3:
                interval_key = '20-30%'
            elif 0.3 <= relative_length < 0.4:
                interval_key = '30-40%'
            else:
                interval_key = '>=40%'
                
            interval_ious[interval_key].append(max(iou_, 0))
            interval_recalls[interval_key] += thresh <= iou_
            interval_counts[interval_key] += 1
        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))
    
    # 구간별 mIoU와 recall 계산 및 출력
    for interval, interval_ious_list in interval_ious.items():
        if interval_ious_list:  # 구간에 IoU 값이 존재하는 경우에만 계산
            mIoU = sum(interval_ious_list) / len(interval_ious_list)
            interval_recall = interval_recalls[interval] / interval_counts[interval] if interval_counts[interval] > 0 else np.array([0, 0, 0])
            print(f'{interval} - mIoU: {mIoU}, Recall: {interval_recall}')
        else:
            print(f'{interval} - mIoU: No data, Recall: No data')


def eval(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        video_feature_path = os.path.join(feature_path, vid+'.npy')
        video_feature = np.load(video_feature_path)
        if pad_sec > 0:
            pad_noise = np.random.randn(round(video_feature.shape[0] / duration * pad_sec), video_feature.shape[1], video_feature.shape[2])
            video_feature = np.concatenate([pad_noise, video_feature], axis=0)
            duration += pad_sec

        for i in range(len(ann['sentences'])):
            query_json = [{'descriptions': ann['sentences'][i], 'gt': ann['timestamps'][i], 'duration': duration}]
            proposals = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), gamma=0.2, cand_num=12, kmeans_k=7)
            s, e = ann['timestamps'][i]
            s, e = s + pad_sec, e + pad_sec

            sp, ep = proposals[0][0],  proposals[0][1]
            # import pdb;pdb.set_trace()
            iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))


def eval_qvhighlight(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    submission = []
    for ann in tqdm(data):
        vid = ann['vid']
        duration = ann['duration']
        query_json = [{'descriptions': [ann['query']]}]

        duration = ann['duration']
        video_feature_path = os.path.join(feature_path, vid+'.npy')
        video_feature = np.load(video_feature_path)
        if pad_sec > 0:
            pad_noise = np.random.randn(round(video_feature.shape[0] / duration * pad_sec), video_feature.shape[1], video_feature.shape[2])
            video_feature = np.concatenate([pad_noise, video_feature], axis=0)
            duration += pad_sec

        ans = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
        submission.append({
            "qid": ann['qid'],
            "query": ann['query'],
            "vid": vid,
            "pred_relevant_windows": [[p['start'], p['end'], p['confidence']] for p in ans[0]['response'][:7]], 
        })
    results = eval_submission(submission, data, verbose=True, match_number=False)
    print(json.dumps(results['brief'], indent=4))


if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    if args.dataset == 'qvhighlight':
        with open(dataset['splits'][args.split]['annotation_file']) as f:
            data = f.readlines()
        data = [json.loads(d) for d in data]
        eval_qvhighlight(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
    else:
        if args.llm_output and os.path.exists(args.llm_output):
            with open(args.llm_output) as f:
                data = json.load(f)
            eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        else:
            with open(dataset['splits'][args.split]['annotation_file']) as f:
                data = json.load(f)
            eval(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        
