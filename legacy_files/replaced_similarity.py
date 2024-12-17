from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from vlm_localizer import localize, calc_scores
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


def eval_with_llm(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    # 유사도 저장용 딕셔너리
    all_similarities = {}
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['revise_and_replace'])):

            query_json = [{'descriptions': ann['revise_and_replace'][i]["revised_query"][0]}]
            replaced_query_json = ann['masked'][i]
            answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor))
            proposals = []
            for t in range(3):
                proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
            
            proposals = proposals[:7]
            proposals = select_proposal(np.array(proposals))
            
            gt = ann['timestamps'][i]
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_
            
            scores = {}
            revised_query_score = calc_scores(video_feature, [ann['revise_and_replace'][i]["revised_query"][0]]).squeeze().cpu().numpy().tolist()
            scores['revised_query'] = revised_query_score

            for replaced_element, replaced_queries in replaced_query_json.items():
                scores[replaced_element] = []
                for replaced_query in replaced_queries:
                    if len(replaced_query) == 0:
                        continue
                    replaced_query_score = calc_scores(video_feature, [replaced_query]).squeeze().cpu().numpy().tolist()
                    scores[replaced_element].append(replaced_query_score)
            
            similarity_entry = {
                "sentence_index": i,
                "sentence": ann['revise_and_replace'][i]["revised_query"][0],
                "subject_masked": replaced_query_json.get('subject_masked', [None]),
                "verb_masked": replaced_query_json.get('verb_masked', [None]),
                "object_masked": replaced_query_json.get('object_masked', [None]),
                "prepositional_phrase_masked": replaced_query_json.get('prepositional_phrase_masked', [None]),
                "proposal": proposals[:1].tolist(),
                "duration": duration,
                "timestamps": ann.get("timestamps", [])[i],
                "similarity_scores": scores,
            }

            # vid가 이미 all_similarities에 있는지 확인하고, 없으면 초기화
            if vid not in all_similarities:
                all_similarities[vid] = []

            # 유사도 정보를 리스트에 추가
            all_similarities[vid].append(similarity_entry)

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))
    
    # 유사도 파일로 저장
    with open("similarities_per_frame-charades-masked-indices_correct.json", "w") as f:
        json.dump(all_similarities, f, indent=4)

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    
    with open(args.llm_output) as f:
        data = json.load(f)
    eval_with_llm(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])
