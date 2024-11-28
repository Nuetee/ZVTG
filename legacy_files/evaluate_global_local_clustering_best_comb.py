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


def eval_with_llm(data, dataset_name, feature_path, stride, max_stride_factor):
    best_recall = np.array([0, 0, 0])
    best_miou = 0
    
    kmeans_k = 4
    if dataset_name == 'charades':
        temporal_window_size = 21
        gamma_list = [0.2]
        cand_num_list = [6, 7, 8, 9, 10, 11, 12]
        prior_list = [0.5, 0.6, 0.7]

    elif dataset_name == 'activitynet':
        temporal_window_size = 25
        gamma_list = [0.2, 0.4, 0.8]
        cand_num_list = [15, 16, 17, 18, 19, 20, 21, 22]
        prior_list = [1]
    
    best_gamma = 0
    best_cand_num = 0
    best_prior = 0

    for gamma in gamma_list:
        for cand_num in cand_num_list:
            for prior in prior_list:
                ious = []
                thresh = np.array([0.3, 0.5, 0.7])
                recall = np.array([0, 0, 0])
                print(f"Dataset: {dataset_name}, (k, gamma, cand_num, prior, temporal window size): ({kmeans_k}, {gamma}, {cand_num}, {prior}, {temporal_window_size})")
                pbar = tqdm(data.items())
                for vid, ann in pbar:
                    duration = ann['duration']
                    video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
                    num_frames = video_feature.shape[0]
                    for i in range(len(ann['sentences'])):
                        # query
                        query_json = [{'descriptions': ann['sentences'][i], 'gt': ann['timestamps'][i], 'duration': ann['duration']}]
                        proposals = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), gamma, cand_num, kmeans_k, prior, temporal_window_size, use_llm=False)
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
                    best_gamma = gamma
                    best_cand_num = cand_num
                    best_prior = prior
            
                print(f'mIoU-{gamma}/{cand_num}/{prior}: {current_miou}')
                for th, r in zip(thresh, recall):
                    print(f'R@{th}:', r / len(ious))

    # 최적의 조합 출력
    print(f"K: {kmeans_k}, Temporal window size: {temporal_window_size}")
    print('Best mIoU:', best_miou)
    for th, r in zip(thresh, best_recall):
        print(f'R@{th}:', r / len(ious))
    print(f'Best gamma: {best_gamma}')
    print(f'Best cand num: {best_cand_num}')
    print(f'Best prioir: {best_prior}')

def eval(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
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
            eval_with_llm(data, args.dataset, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'])
        else:
            with open(dataset['splits'][args.split]['annotation_file']) as f:
                data = json.load(f)
            eval(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        
