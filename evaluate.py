from data_configs import DATASETS
import argparse
import numpy as np
from chat_bots import get_chat_model
import json
from tqdm import tqdm
from vlm_localizer import localize
from qvhhighlight_eval import eval_submission
import os
from llm_prompting import select_proposal, filter_and_integrate
import pdb
import time
import itertools

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use only VLM for evaluation.')
    parser.add_argument('--api', action='store_true', help='Enable use GPT API call')

    return parser.parse_args()


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union


def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    
    gt_lengths = []

    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            # sub queries
            sub_query_proposals = []
            if 'query_json' in ann['response'][i]:
                relation = ann['response'][i]['relationship']
                # j의 range가 1부터 시작하는 이유는 0번째는 sub-query가 아닌 전체 query이기 때문
                for j in range(1, len(ann['response'][i]['query_json'])):
                    query_json = [{'descriptions': q} for q in ann['response'][i]['query_json'][j]['descriptions']]
                    # 하나의 description에 대해 10개 이하의 response(st:end, confidence) / 10개 이하인 이유는 10개를 뽑지만 nms에 의해 억제된 경우 그 이하의 proposal들이 반환되기 때문
                    answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
                    proposals = []
                    # 각 description에 대한 response에서 상위 3개만 proposal에 저장 -> proposals에는 총 9개의 구간 저장
                    for t in range(3):
                        proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
                    proposals = np.array(proposals)
                    # 하나의 sub-query에 대해서 3개의 proposal을 선택
                    sub_query_proposals.append(select_proposal(proposals)[:3])
            else:
                relation = 'single-query'

            # query, 원문 쿼리 하나 + llm 생성 description 3개 => 4개의 description
            query_json = [{'descriptions': ann['sentences'][i]}]
            if 'query_json' in ann['response'][i]:
                query_json += [{'descriptions': q} for q in ann['response'][i]['query_json'][0]['descriptions']]
            answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
            proposals = []
            for t in range(3):
                proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
            
            # 총 12개의 proposals에서 앞 7개의 proposals 가져옴 -> 각 description 별 1개씩 + 3개
            proposals = proposals[:7] + filter_and_integrate(sub_query_proposals, relation)

            # select_proposal은 각 proposal에 대해서, ((자기 자신과 다른 proposal과의 겹침 정도)^gamma * 다른 proposal의 score)의 합을 계산하여 정렬됨. 따라서, 점수가 높은 proposal과 많이 겹칠수록 높은 점수를 가짐
            proposals = select_proposal(np.array(proposals))
            
            gt = ann['timestamps'][i]
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

            gt_lengths.append(gt[1] - gt[0])

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))


def eval_with_api(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    bot = get_chat_model(model_name="gpt-4-turbo", api_key='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')
    
    gt_lengths = []

    # pbar = tqdm(data.items())
    pbar = tqdm(itertools.islice(data.items(), 100))
    start_time = time.time()  # 실행 시간 측정 시작

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            query = ann['sentences'][i]
            try:
                ans_json, raw = bot.ask(query)
                succ = True
            except Exception as exp:
                print(exp)
                if isinstance(exp, KeyboardInterrupt):
                    exit()

            sub_query_proposals = []
            if ans_json:
                relation = ans_json['relationship']
                for j in range(1, len(ans_json['query_json'])):
                    query_json = [{'descriptions': q} for q in ans_json['query_json'][j]['descriptions']]
                    answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
                    proposals = []
                    
                    for t in range(3):
                        proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
                    proposals = np.array(proposals)
                    sub_query_proposals.append(select_proposal(proposals)[:3])
            else:
                relation = 'single-query'

            query_json = [{'descriptions': ann['sentences'][i]}]
            if 'query_json' in ann['response'][i]:
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

            gt_lengths.append(gt[1] - gt[0])

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    elapsed_time = time.time() - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")
    
    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))


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
            if args.api:
                eval_with_api(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
            else:
                eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
        else:
            with open(dataset['splits'][args.split]['annotation_file']) as f:
                data = json.load(f)
            eval(data, dataset['feature_path'], dataset['splits'][args.split]['pad_sec'])
        
