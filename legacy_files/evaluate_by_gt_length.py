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

def eval_with_llm(data, dataset_name, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
    if dataset_name == "charades":
        interval = [5,10,15,20]
        interval_ious = {'<5s': [], '<10s': [], '<15s': [], '<20s': []}
        interval_recalls = {'<5s': np.array([0, 0, 0]), '<10s': np.array([0, 0, 0]), '<15s': np.array([0, 0, 0]), '<20s': np.array([0, 0, 0])}
        interval_counts = {'<5s': 0, '<10s': 0, '<15s': 0, '<20s': 0}  # 각 구간의 개수를 기록
    elif dataset_name == 'activitynet':
        interval = [30,60,90,120]
        interval_ious = {'<30s': [], '<60s': [], '<90s': [], '<120s': []}
        interval_recalls = {'<30s': np.array([0, 0, 0]), '<60s': np.array([0, 0, 0]), '<90s': np.array([0, 0, 0]), '<120s': np.array([0, 0, 0])}
        interval_counts = {'<30s': 0, '<60s': 0, '<90s': 0, '<120s': 0}  # 각 구간의 개수를 기록
    
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
            # query
            query_json = [{'descriptions': ann['sentences'][i]}]
            if 'query_json' in ann['response'][i] and len(ann['response'][i]['query_json']) > 0:
                query_json += [{'descriptions': q} for q in ann['response'][i]['query_json'][0]['descriptions']]
            answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
            proposals = []
            for t in range(3):
                proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
            proposals = proposals[:7] + filter_and_integrate(sub_query_proposals, relation)
            proposals = select_proposal(np.array(proposals))
                
            gt = ann['timestamps'][i]
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

            # gt의 상대적 길이에 따른 비율 계산 및 구간별 IoU와 recall 기록
            gt_length = gt[1] - gt[0] 
            if gt_length < interval[0]:
                interval_key = '<' + str(interval[0]) + 's'
                interval_ious[interval_key].append(max(iou_, 0))
                interval_recalls[interval_key] += thresh <= iou_
                interval_counts[interval_key] += 1
            if gt_length < interval[1]:
                interval_key = '<' + str(interval[1]) + 's'
                interval_ious[interval_key].append(max(iou_, 0))
                interval_recalls[interval_key] += thresh <= iou_
                interval_counts[interval_key] += 1
            if gt_length < interval[2]:
                interval_key = '<' + str(interval[2]) + 's'
                interval_ious[interval_key].append(max(iou_, 0))
                interval_recalls[interval_key] += thresh <= iou_
                interval_counts[interval_key] += 1
            if gt_length < interval[3]:
                interval_key = '<' + str(interval[3]) + 's'
                interval_ious[interval_key].append(max(iou_, 0))
                interval_recalls[interval_key] += thresh <= iou_
                interval_counts[interval_key] += 1

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('Overall mIoU:', sum(ious) / len(ious))
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

def eval(data, feature_path, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    stride = 20
    max_stride_factor = 0.5
    
    pbar = tqdm(data.items())
    for vid, ann in pbar:
        query_json = []
        for i in range(len(ann['sentences'])):
            query_json.append({'descriptions': [ann['sentences'][i]]})

        duration = ann['duration'] if 'duration' in ann else ann['video_duration']
        video_feature_path = os.path.join(feature_path, vid+'.npy')
        video_feature = np.load(video_feature_path)
        if pad_sec > 0:
            pad_noise = np.random.randn(round(video_feature.shape[0] / duration * pad_sec), video_feature.shape[1], video_feature.shape[2])
            video_feature = np.concatenate([pad_noise, video_feature], axis=0)
            duration += pad_sec

        ans = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
        for i in range(len(ans)):
            s, e = ann['timestamps'][i]
            s, e = s + pad_sec, e + pad_sec

            sp, ep = ans[i]['response'][0]['start'], ans[i]['response'][0]['end']
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
            eval_with_llm(data, args.dataset, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        else:
            with open(dataset['splits'][args.split]['annotation_file']) as f:
                data = json.load(f)
            eval(data, dataset['feature_path'], dataset['splits'][args.split]['pad_sec'])
        
