from data_configs import DATASETS
import argparse
import numpy as np
import json
from tqdm import tqdm
from vlm_localizer_TAG import localize
from qvhhighlight_eval import eval_submission
import os
from llm_prompting import select_proposal, filter_and_integrate
import pdb

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use only VLM for evaluation.')
    parser.add_argument('--kmeans_gpu', action='store_true', help='Enable use GPU KMeans')

    return parser.parse_args()


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union


def compute_ious(proposals, ground_truth):
    """
    주어진 모든 proposals과 단일 Ground Truth 구간 간의 IoU를 벡터 연산으로 계산.

    :param proposals: (N, 3) 형태의 numpy 배열 (start, end, score)
    :param ground_truth: 1차원 리스트 [start, end] 형태의 단일 GT 구간
    :return: (N,) 형태의 numpy 배열 (각 proposal과 ground truth 간의 IoU 값)
    """
    gt_start, gt_end = ground_truth  # Ground Truth의 start, end 추출

    # proposals의 start, end 추출
    p_start = proposals[:, 0]  # (N,)
    p_end = proposals[:, 1]    # (N,)

    # 교집합 계산
    inter_start = np.maximum(p_start, gt_start)  # (N,)
    inter_end = np.minimum(p_end, gt_end)        # (N,)
    inter_length = np.maximum(0, inter_end - inter_start)  # 음수 방지 (N,)

    # 합집합 계산
    p_length = p_end - p_start  # (N,)
    gt_length = gt_end - gt_start  # 스칼라 값
    union_length = p_length + gt_length - inter_length  # (N,)

    # IoU 계산 (분모가 0이 되지 않도록 예외 처리)
    iou_values = np.where(union_length > 0, inter_length / union_length, 0)

    return iou_values  # (N,) 크기의 IoU 배열 반환

def compute_proposal_iou_matrix(proposals):
    """
    모든 proposals 간의 IoU를 벡터 연산으로 계산하여 IoU 행렬을 반환.

    :param proposals: (N, 3) 형태의 numpy 배열 (start, end, score)
    :return: (N, N) 형태의 IoU 행렬
    """
    p_start = proposals[:, None, 0]  # (N, 1) -> (N, N) 형태 확장
    p_end = proposals[:, None, 1]    # (N, 1) -> (N, N) 형태 확장

    q_start = proposals[:, 0]  # (N,)
    q_end = proposals[:, 1]    # (N,)

    # 교집합 계산
    inter_start = np.maximum(p_start, q_start)  # (N, N)
    inter_end = np.minimum(p_end, q_end)        # (N, N)
    inter_length = np.maximum(0, inter_end - inter_start)  # 음수 방지 (N, N)

    # 합집합 계산
    p_length = p_end - p_start  # (N, 1)
    q_length = q_end - q_start  # (N,)
    union_length = p_length + q_length - inter_length  # (N, N)

    # IoU 계산 (분모가 0이 되지 않도록 예외 처리)
    iou_matrix = np.where(union_length > 0, inter_length / union_length, 0)

    return iou_matrix  # (N, N) 크기의 IoU 행렬 반환


def eval_TAG(data, feature_path, stride, hyperparams, kmeans_gpu):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    proposal_len = []
    max_iou_list = []
    proposal_iou_mean_list = []
    gt_nearest_proposal_ratio_list = []
    gt_nearest_proposal_iou_list = []

    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        
        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            gt_len = gt[1] - gt[0]
            near_start = max(0, gt[0] - gt_len)
            near_end = min(duration, gt[1] + gt_len)


            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, duration, query_json, stride, hyperparams, kmeans_gpu)
            proposals = proposals[:6]

            near_proposals = [p for p in proposals if p[0] >= near_start and p[1] <= near_end]
            gt_nearest_proposal_ratio_list.append(len(near_proposals) / len(proposals))
            
            if len(near_proposals) > 0:
                near_iou_values = compute_ious(np.array(near_proposals), np.array(gt))
                gt_nearest_proposal_iou_list.append(np.mean(near_iou_values))

            proposals = select_proposal(np.array(proposals))
            proposal_len.append(len(proposals))
            
            iou_matrix = compute_proposal_iou_matrix(proposals)
            np.fill_diagonal(iou_matrix, 0)
            
            iou_values = compute_ious(proposals, np.array(gt))
            max_iou_list.append(np.max(iou_values))

            # IoU 합과 평균 계산 (자기 자신 제외)
            num_elements = (iou_matrix.shape[0] * iou_matrix.shape[1]) - len(proposals)
            iou_mean = np.sum(iou_matrix) / num_elements if num_elements > 0 else 0  # 평균 계산
            proposal_iou_mean_list.append(iou_mean)

            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

    print('Mean Max IoU: ', sum(max_iou_list) / len(max_iou_list))
    print('Mean proposal IoU mean: ', sum(proposal_iou_mean_list) / len(proposal_iou_mean_list))
    print('Mean # of near proposals with gt: ', sum(gt_nearest_proposal_ratio_list) / len(gt_nearest_proposal_ratio_list))
    print('mIoU of near proposals: ', sum(gt_nearest_proposal_iou_list) / len(gt_nearest_proposal_iou_list))

    print(sum(proposal_len) / len(proposal_len))
    print(min(proposal_len))
    print(max(proposal_len))
    print(sum(1 for x in proposal_len if x >= 5))
    print(len(proposal_len))


if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    
    with open(args.llm_output) as f:
        data = json.load(f)

    eval_TAG(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args.kmeans_gpu)


        
