from data_configs import DATASETS
import argparse
import numpy as np
from chat_bots import get_chat_model
import json
from tqdm import tqdm
from vlm_localizer_ablation import localize
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


def relative_distance(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    gt_len = e - s
    rel_d1 = abs(start - s) / gt_len
    rel_d2 = abs(end - e) / gt_len
    return (rel_d1 + rel_d2) / 2


def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    pbar = tqdm(data.items())
    reldiss = []
    total_proposal_count = 0
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        proposal_count_flag = True
        for i in range(len(ann['sentences'])):
            # query, 원문 쿼리 하나 + llm 생성 description 3개 => 4개의 description
            query_json = [{'descriptions': ann['sentences'][i]}]
            answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
            proposals = []

            for t in range(len(answers[0]['response'])):
                proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]

            if proposal_count_flag:
                total_proposal_count += len(proposals)
                proposal_count_flag = False

            proposals = np.array(proposals)
            gt = ann['timestamps'][i]
            best_iou = 0
            best_reldis = float('inf')
            for i in range(len(proposals)):
                iou_ = calc_iou(proposals[i:i+1], gt)[0]
                reldis_ = relative_distance(proposals[i:i+1], gt)[0]
                if iou_ > best_iou:
                    best_iou = iou_
                if reldis_ < best_reldis:
                    best_reldis = reldis_

            ious.append(max(best_iou, 0))
            recall += thresh <= best_iou
            reldiss.append(min(best_reldis, float('inf')))

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), "mIoU": sum(reldiss) / len(reldiss), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    print('Rel.dis:', sum(reldiss) / len(reldiss))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))
    print(f'Total proposals: {total_proposal_count}, Mean proposals per video: {total_proposal_count / len(data.items())}')

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
        
