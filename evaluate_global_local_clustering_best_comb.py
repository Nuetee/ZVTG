from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer_global_local_clustering_left_skewed_distribution import localize_best_comb
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


def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    max_ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    max_recall = np.array([0, 0, 0])
    best_miou = 0
    best_alpha = 1

    # 그리드 서치를 위한 파라미터 범위
    alpha = np.linspace(1, 0, 10)

    for current_alpha in alpha:
        pbar = tqdm(data.items())
        for vid, ann in pbar:
            duration = ann['duration']
            video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
            num_frames = video_feature.shape[0]
            for i in range(len(ann['sentences'])):
                # query
                query_json = [{'descriptions': ann['sentences'][i], 'gt': ann['timestamps'][i], 'duration': ann['duration']}]
                proposals = localize_best_comb(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), gamma=0.4, alpha=current_alpha)
                gt = ann['timestamps'][i]
                iou_ = calc_iou(proposals[:1], gt)[0]
                ious.append(max(iou_, 0))
                recall += thresh <= iou_

                ######
                max_iou_ = 0
                for i in range(len(proposals)):
                    temp_iou_ = calc_iou(proposals[i:i+1], gt)[0]
                    if temp_iou_ > max_iou_:
                        max_iou_ = temp_iou_
                max_ious.append(max(max_iou_, 0))
                max_recall += thresh <= max_iou_
                ######

            # pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})
            pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious)), "max_mIoU": sum(max_ious) / len(max_ious), 'max_recall': str(max_recall / len(max_ious))})

        # mIoU 계산
        current_miou = sum(ious) / len(ious) if len(ious) > 0 else 0
        if current_miou > best_miou:
            best_miou = current_miou
            best_alpha = current_alpha
        print(f'mIoU - {current_alpha}: {current_miou}')

    # 최적의 조합 출력
    print('Best mIoU:', best_miou)
    print('Best Alpha:', best_alpha)



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
            eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        else:
            with open(dataset['splits'][args.split]['annotation_file']) as f:
                data = json.load(f)
            eval(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        
