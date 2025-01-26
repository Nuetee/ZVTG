from data_configs import DATASETS
import argparse
import numpy as np
import json
import torch
from tqdm import tqdm
from chat_bots import get_chat_model
from vlm_localizer_TAG_ablation import localize
from qvhhighlight_eval import eval_submission
import os
from llm_prompting import select_proposal, select_proposal_with_score
import time
import itertools

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--use_llm', action='store_true', help='Enable use llm')
    parser.add_argument('--kmeans_gpu', action='store_true', help='Enable use GPU KMeans')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use nonly VLM for evaluation.')
    parser.add_argument('--api', action='store_true', help='Enable use GPT API call')

    return parser.parse_args()


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

from collections import defaultdict
def eval_without_llm(data, feature_path, stride, hyperparams, kmeans_gpu):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    pbar = tqdm(data.items())
    # pbar = tqdm(itertools.islice(data.items(), 100))
    # start_time = time.time()  # 실행 시간 측정 시작

    mean_time_variance_list = []
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        kmeans_labels_flag = True
        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals, kmeans_labels = localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)
            if kmeans_labels_flag:
                frame_indices = torch.arange(len(kmeans_labels))
                unique_labels = kmeans_labels.unique()  # 고유 클러스터 라벨
                time_variances = []  # 각 클러스터의 시간 분산 저장

                for label in unique_labels:
                    cluster_time_indices = frame_indices[kmeans_labels == label]  # 해당 클러스터의 시간 인덱스 추출
                    if len(cluster_time_indices) > 1:  # 데이터가 2개 이상 있는 경우에만 분산 계산
                        variance = torch.var(cluster_time_indices.float())
                    else:
                        variance = 0  # 데이터가 없거나 하나인 경우 기본값 할당
                    time_variances.append(variance)

                # 클러스터별 시간 분산 평균 계산
                mean_time_variance = torch.mean(torch.tensor(time_variances))
                mean_time_variance_list.append(mean_time_variance.item())
                kmeans_labels_flag = False
            # proposals = select_proposal(np.array(proposals))

            # iou_ = calc_iou(proposals[:1], gt)[0]
            # ious.append(max(iou_, 0))
            # recall += thresh <= iou_

        pbar.set_postfix({"mTime var.": sum(mean_time_variance_list) / len(mean_time_variance_list)})

    # elapsed_time = time.time() - start_time
    # print(f"Execution Time: {elapsed_time:.2f} seconds")

    # print('mIoU:', sum(ious) / len(ious))
    # for th, r in zip(thresh, recall):
    #     print(f'R@{th}:', r / len(ious))
    print(sum(mean_time_variance_list)/len(mean_time_variance_list))


def eval_with_llm(data, feature_path, stride, hyperparams, kmeans_gpu):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    pbar = tqdm(data.items())
    
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)
            
            if 'query_json' in ann['response'][i]:
                for j in range(len(ann['response'][i]['query_json'][0]['descriptions'])):
                    query_json = [{'descriptions': ann['response'][i]['query_json'][0]['descriptions'][j]}]
                    proposals += localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)

            proposals = select_proposal(np.array(proposals))
        
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))



def eval_with_api(data, feature_path, stride, hyperparams, kmeans_gpu):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    bot = get_chat_model(model_name="gpt-4-turbo", api_key='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')

    # pbar = tqdm(data.items())
    pbar = tqdm(itertools.islice(data.items(), 100))
    start_time = time.time()  # 실행 시간 측정 시작
    
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)
            
            query = ann['sentences'][i]
            try:
                ans_json, raw = bot.ask(query)
                succ = True
            except Exception as exp:
                print(exp)
                if isinstance(exp, KeyboardInterrupt):
                    exit()

            if ans_json:
                for j in range(len(ans_json['query_json'][0]['descriptions'])):
                    query_json = [{'descriptions': ans_json['query_json'][0]['descriptions'][j]}]
                    proposals += localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)

            proposals = select_proposal(np.array(proposals))
        
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})
    
    elapsed_time = time.time() - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))


def eval(data, feature_path, stride, hyperparams, use_llm, kmeans_gpu, pad_sec=0.0):
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
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)
            
            if use_llm:
                if 'query_json' in ann['response'][i]:
                    for j in range(len(ann['response'][i]['query_json'][0]['descriptions'])):
                        query_json = [{'descriptions': ann['response'][i]['query_json'][0]['descriptions'][j]}]
                        proposals += localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)

            proposals = select_proposal(np.array(proposals))

            s, e = ann['timestamps'][i]
            s, e = s + pad_sec, e + pad_sec

            sp, ep = proposals[0][0],  proposals[0][1]

            iou_ = (min(e, ep) - max(s, sp)) / (max(e, ep) - min(s, sp))
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))


def eval_qvhighlight(data, feature_path, stride, hyperparams, kmeans_gpu):
    submission = []
    for ann in tqdm(data):
        vid = ann['vid']
        duration = ann['duration']
        query_json = [{'descriptions': [ann['query']]}]

        duration = ann['duration']
        video_feature_path = os.path.join(feature_path, vid+'.npy')
        video_feature = np.load(video_feature_path)

        proposals = localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)
        proposals, proposal_scores = select_proposal_with_score(np.array(proposals))
        
        submission.append({
            "qid": ann['qid'],
            "query": ann['query'],
            "vid": vid,
            "pred_relevant_windows": [[p[0], p[1], proposal_scores[idx]] for idx, p in enumerate(proposals[:7])], 
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
        eval_qvhighlight(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.kmeans_gpu)
    else:
        if args.llm_output and os.path.exists(args.llm_output):
            with open(args.llm_output) as f:
                data = json.load(f)
            if args.use_llm:
                if args.api:
                    eval_with_api(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.kmeans_gpu)
                else:
                    eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.kmeans_gpu)
            else:
                eval_without_llm(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.kmeans_gpu)
        else:
            with open(dataset['splits'][args.split]['annotation_file']) as f:
                data = json.load(f)
            eval(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.use_llm, args.kmeans_gpu, dataset['splits'][args.split]['pad_sec'])