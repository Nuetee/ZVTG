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
    parser.add_argument('--nms_thresh', default=0.3, type=float)
    parser.add_argument('--ratio', default=1, type=float)
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


def eval_sliding(data, feature_path, stride, max_stride_factor, pad_sec=0.0):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    proposal_len_var_list = []
    # proposal_len = []
    max_iou_list = []
    # proposal_iou_mean_list = []
    # gt_nearest_proposal_ratio_list = []
    # gt_nearest_proposal_iou_list = []
    iou_list = []
    score_var_list = []

    pbar = tqdm(data.items())
    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))

        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            # gt_len = gt[1] - gt[0]
            # near_start = max(0, gt[0] - gt_len * args.ratio)
            # near_end = min(duration, gt[1] + gt_len * args.ratio)

            # sub queries
            query_json = [{'descriptions': ann['sentences'][i]}]
            answers = localize(video_feature, duration, query_json, stride, int(video_feature.shape[0] * max_stride_factor), nms_thresh=args.nms_thresh)
            proposals = []
            # proposal_len.append(len(answers[0]['response']))
            
            for t in range(len(answers[0]['response'])):
                proposals += [[p['response'][t]['start'], p['response'][t]['end'], p['response'][t]['confidence']] for p in answers if len(p['response']) > t]
            
            # near_proposals = [p for p in proposals if p[0] >= near_start and p[1] <= near_end]
            # gt_nearest_proposal_ratio_list.append(len(near_proposals) / len(proposals))
            
            # if len(near_proposals) > 0:
            #     near_iou_values = compute_ious(np.array(near_proposals), np.array(gt))
            #     gt_nearest_proposal_iou_list.append(np.mean(near_iou_values))
                

            # proposals = select_proposal(np.array(proposals))
            # iou_matrix = compute_proposal_iou_matrix(proposals)
            # np.fill_diagonal(iou_matrix, 0)

            # # IoU 합과 평균 계산 (자기 자신 제외)
            # num_elements = (iou_matrix.shape[0] * iou_matrix.shape[1]) - len(proposals)
            # iou_mean = np.sum(iou_matrix) / num_elements if num_elements > 0 else 0  # 평균 계산
            # proposal_iou_mean_list.append(iou_mean)
            proposals = np.array(proposals)
            proposals = np.array(proposals)
            proposal_len = proposals[:, 1] - proposals[:, 0]
            proposal_len_var = np.var(proposal_len)
            proposal_len_var_list.append(proposal_len_var)
            scores = proposals[:, 2]
            score_var = np.var(scores)
            score_var_list.append(score_var)
            iou_values = compute_ious(proposals, np.array(gt))
            iou_list.extend(iou_values.tolist())
            max_iou_list.append(np.max(iou_values))
            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})
    
    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

    print('Mean Max IoU: ', sum(max_iou_list) / len(max_iou_list))

    data = {
        "IoU list": iou_list,
        "Score variance": score_var_list,
        "Proposal length variance": proposal_len_var_list
    }

    # JSON 파일로 저장
    with open(f"TFTVG_{args.dataset}_analysis.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    # import matplotlib.pyplot as plt
    # import matplotlib.pyplot as plt

    # def plot_iou_histogram(iou_values):
    #     """
    #     IoU 값들의 분포를 0.1 단위로 0~1 구간에서 히스토그램으로 그리는 함수.
    #     :param iou_values: IoU 값들의 리스트
    #     """
    #     # bins = np.arange(0, 1.1, 0.1)  # 0.1 단위로 0~1 범위의 bin 생성
    #     max_value = max(iou_values) if iou_values else 1  # 리스트가 비어있지 않을 경우 최대값 설정
    #     bins = np.linspace(0, max_value, 11)  # 0부터 최대값까지 10개 구간으로 나누기
    #     plt.figure(figsize=(8, 6))
    #     plt.hist(iou_values, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    #     plt.xlabel('IoU')
    #     plt.ylabel('Frequency')
    #     plt.title('Distribution of IoU Values')
    #     plt.xticks(bins)  # x축을 0.1 단위로 설정
    #     plt.grid(True, linestyle='--', alpha=0.5)
    #     plt.tight_layout()
    #     plt.savefig(f"{args.dataset} TFVTG")
    #     plt.close()
    # plot_iou_histogram(proposal_len_var_list)
    # print('Mean proposal IoU mean: ', sum(proposal_iou_mean_list) / len(proposal_iou_mean_list))
    # print('Mean # of near proposals with gt: ', sum(gt_nearest_proposal_ratio_list) / len(gt_nearest_proposal_ratio_list))
    # print('mIoU of near proposals: ', sum(gt_nearest_proposal_iou_list) / len(gt_nearest_proposal_iou_list))

    # print(sum(proposal_len) / len(proposal_len))
    # print(min(proposal_len))
    # print(max(proposal_len))
    # print(sum(1 for x in proposal_len if x >= 5))
    # print(len(proposal_len))




if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    
    with open(args.llm_output) as f:
        data = json.load(f)

    eval_sliding(data, dataset['feature_path'], dataset['stride'], dataset['max_stride_factor'], dataset['splits'][args.split]['pad_sec'])

        
