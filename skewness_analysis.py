from data_configs import DATASETS
import argparse
import numpy as np
import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from lavis.models import load_model_and_preprocess
from llm_prompting import select_proposal
from torchvision import transforms
from scipy.stats import skew

#### BLIP-2 Q-Former ####
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])
#### BLIP-2 Q-Former ####

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--split', default='default', type=str, help='Specify the split. See supported splits in data_configs.py.')
    parser.add_argument('--kmeans_gpu', action='store_true', help='Enable use GPU KMeans')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use nonly VLM for evaluation.')

    return parser.parse_args()

def scores_masking(scores, masks):
    # scores의 길이가 3 미만인 경우 initial_mask를 그대로 사용
    if scores.shape[1] < 3:
        masks = masks.squeeze()
    else:
        # 양쪽 끝에 2씩 False로 패딩
        padded_masks = F.pad(masks, (1, 1), mode='constant', value=False)

        # 현재 위치를 기준으로 양옆 2개의 값 기반 Majority voting, 최종 마스크 결과 저장
        final_masks = padded_masks.clone()
        for i in range(2, padded_masks.shape[1] - 1):
            window = padded_masks[:, i - 1 : i + 2]
            if window.sum() < 2:
                final_masks[:, i] = 0

        # 패딩 제거하여 원래 크기의 마스크로 복원
        masks = final_masks[:, 1:-1].squeeze()
    
    # 모든 값이 False일 경우 전부 True로 설정
    if not masks.any():
        masks[:] = True

    # final_mask를 기반으로 masked_indices 계산
    masked_indices = torch.nonzero(masks, as_tuple=True)[0]  # 마스킹된 실제 인덱스 저장
    
    return masks, masked_indices


def alignment_adjustment(data, scale_gamma, device, lambda_max=2, lambda_min=-2):
    # 작은 상수 추가로 양수 데이터 보장
    epsilon = 1e-6
    data = data + abs(data.min()) + epsilon if np.any(data <= 0) else data
    
    def boxcox_transformed(x, lmbda):
        if lmbda == 0:
            return np.log(x)
        else:
            return (x**lmbda - 1) / lmbda

    # 최적의 lambda를 찾기 위한 로그 가능도 함수 (최소화할 함수)
    def neg_log_likelihood(lmbda):
        transformed_data = boxcox_transformed(data, lmbda)
        # 분산 계산 시 overflow 방지
        var = np.var(transformed_data, ddof=1)
        return -np.sum(np.log(np.abs(transformed_data))) + 0.5 * len(data) * np.log(var)

    # lambda 범위 내에서 최적화
    result = minimize_scalar(neg_log_likelihood, bounds=(lambda_min, lambda_max), method='bounded')
    best_lambda = result.x
    
    # 최적의 lambda로 변환 데이터 생성
    transformed_data = boxcox_transformed(data, best_lambda)

    original_min, original_max = data.min(), data.max()
    transformed_min, transformed_max = transformed_data.min(), transformed_data.max()
    transformed_data = (transformed_data - transformed_min) / (transformed_max - transformed_min)  # normalize to [0, 1]
    is_scale = False
    if original_max - original_min > scale_gamma:
        is_scale = True
        transformed_data = transformed_data * (original_max - original_min) + original_min  # scale to original min/max
    else:
        transformed_data = transformed_data * (scale_gamma) + original_min
    # 변환 결과를 다시 텐서로 변환하고 원래 형태로 복원

    normalized_scores = torch.tensor(transformed_data, device=device).unsqueeze(0)

    return normalized_scores, is_scale


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union


def eval_without_llm(data, feature_path, stride, hyperparams, args):
    pbar = tqdm(data.items())

    skew_tuples = []  # (vid, sentence_idx, data_skewness, delta_skewness)

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        num_frames = video_feature.shape[0]

        data_skews = []
        norm_skews = []
        deltas = []

        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            sentence = ann['sentences'][i]

            with torch.no_grad():
                text = model.tokenizer(sentence, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
                text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
                text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
            v1 = F.normalize(text_feat, dim=-1)
            v2 = F.normalize(torch.tensor(video_feature, device='cuda', dtype=v1.dtype), dim=-1)
            scores = torch.einsum('md,npd->mnp', v1, v2)
            scores, scores_idx = scores.max(dim=-1)
            scores = scores.mean(dim=0, keepdim=True)

            initial_masks = (scores > 0.2 if hyperparams['is_blip2'] else scores > 0)
            masks, masked_indices = scores_masking(scores, initial_masks)

            data_np = scores[:, masks].flatten().cpu().numpy()
            data_skew = skew(data_np)

            normalized_scores, _ = alignment_adjustment(data_np, hyperparams['gamma'], scores.device)
            norm_skew = skew(normalized_scores[0].cpu().numpy())

            delta = data_skew - norm_skew
            data_skews.append(data_skew)
            norm_skews.append(norm_skew)
            deltas.append(delta)

            skew_tuples.append((vid, i, data_skew, delta))

        # 원래 데이터에 skewness 정보 저장
        ann["skewness_meta"] = {
            "data_skewness": data_skews,
            "normalized_skewness": norm_skews,
            "delta_skewness": deltas
        }

    # 전체 데이터를 왜도 절대값 기준 정렬
    skew_tuples.sort(key=lambda x: abs(x[2]), reverse=True)  # 또는 x[3] for delta

    total = len(skew_tuples)
    q1 = skew_tuples[:total // 3]
    q2 = skew_tuples[total // 3: 2 * total // 3]
    q3 = skew_tuples[2 * total // 3:]
    # q4 = skew_tuples[3 * total // 4:]

    def filter_data(original_data, target_tuples):
        filtered = {}
        for vid, idx, _, _ in target_tuples:
            if vid not in filtered:
                filtered[vid] = {
                    "duration": original_data[vid]["duration"],
                    "sentences": [],
                    "timestamps": [],
                    "skewness_meta": {
                        "data_skewness": [],
                        "normalized_skewness": [],
                        "delta_skewness": []
                    }
                }
            filtered[vid]["sentences"].append(original_data[vid]["sentences"][idx])
            filtered[vid]["timestamps"].append(original_data[vid]["timestamps"][idx])
            for key in ["data_skewness", "normalized_skewness", "delta_skewness"]:
                filtered[vid]["skewness_meta"][key].append(original_data[vid]["skewness_meta"][key][idx])
        return filtered

    # 저장 디렉토리 및 결과 파일 생성
    os.makedirs("skewness_analysis", exist_ok=True)
    json.dump(filter_data(data, q1), open(f"skewness_analysis/{args.dataset}_top33_data_skew.json", "w"), indent=2)
    json.dump(filter_data(data, q2), open(f"skewness_analysis/{args.dataset}_top33to66_data_skew.json", "w"), indent=2)
    json.dump(filter_data(data, q3), open(f"skewness_analysis/{args.dataset}_top66to100_data_skew.json", "w"), indent=2)
    # json.dump(filter_data(data, q4), open(f"skewness_analysis/{args.dataset}_bottom25_data_skew.json", "w"), indent=2)
    json.dump(data, open(f"skewness_analysis/{args.dataset}_full_data_with_skewness.json", "w"), indent=2)

    print("✅ 저장 완료: skewness 기반 데이터 분할 및 전체 메타 포함")
       


if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
    assert args.split in dataset['splits'], 'Unsupported split. To evaluate other split, please add the configuration in data_configs.py.'
    
    print('Evaluating', args.dataset, args.split)
    with open(args.llm_output) as f:
        data = json.load(f)
    eval_without_llm(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'], args)