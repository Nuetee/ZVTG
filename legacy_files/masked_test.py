from data_configs import DATASETS
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use only VLM for evaluation.')

    return parser.parse_args()


def mask_tokens(tokenizer, input_ids, mask_prob=0.1):
    """
    의미 있는 토큰 중 일부를 마스킹하되, PAD, CLS, SEP, 그리고 '.', ','를 제외.
    - tokenizer: 모델의 토크나이저
    - input_ids: 입력 텍스트 토큰 ID (Tensor)
    - mask_prob: 마스킹 확률 (default: 0.1)
    """
    device = input_ids.device
    masked_input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    # PAD, CLS, SEP 토큰 제외
    special_tokens = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    special_tokens_ids = set(special_tokens)

    # '.', ','에 해당하는 토큰 ID 가져오기
    dot_token_id = tokenizer.convert_tokens_to_ids('.')
    comma_token_id = tokenizer.convert_tokens_to_ids(',')

    # 마스킹 대상 토큰 필터링
    for i in range(batch_size):
        valid_indices = []
        for j in range(seq_len):
            token_id = input_ids[i, j].item()
            if token_id not in special_tokens_ids and token_id != dot_token_id and token_id != comma_token_id:
                valid_indices.append(j)  # 유효 토큰만 추가

        # 유효 토큰 중 일부를 마스킹 (최소 1개 보장)
        if valid_indices:
            num_to_mask = max(1, int(mask_prob * len(valid_indices)))  # 최소 1개는 마스킹
            mask_indices = torch.randperm(len(valid_indices))[:num_to_mask]
            for idx in mask_indices:
                masked_input_ids[i, valid_indices[idx]] = tokenizer.mask_token_id

    return masked_input_ids


import random
def eval_test2(data, feature_path, stride, hyperparams):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    # # Shuffle the data keys
    # shuffled_data = list(data.items())
    # random.shuffle(shuffled_data)
    # pbar = tqdm(shuffled_data)

    pbar = tqdm(data.items())

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        
        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]

            with torch.no_grad():
                text = model.tokenizer(ann['sentences'][i], padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')

                text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
                text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
            v1 = F.normalize(text_feat, dim=-1)
            v2 = F.normalize(torch.tensor(video_feature, device='cuda', dtype=v1.dtype), dim=-1)
            scores = torch.einsum('md,npd->mnp', v1, v2)
            scores, scores_idx = scores.max(dim=-1)
            scores = scores.mean(dim=0, keepdim=True)
            
            scores_1d = scores.squeeze()

            verb_masked = ann['masked'][i]["verb_masked"]
            if len(verb_masked) == 0:
                continue
            with torch.no_grad():
                masked_text = model.tokenizer(verb_masked[0], padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
                masked_text_output = model.Qformer.bert(masked_text.input_ids, attention_mask=text.attention_mask, return_dict=True)
                masked_text_feat = model.text_proj(masked_text_output.last_hidden_state[:, 0, :])
            v1 = F.normalize(masked_text_feat, dim=-1)
            masked_scores = torch.einsum('md,npd->mnp', v1, v2)
            masked_scores, _ = masked_scores.max(dim=-1)
            masked_scores = masked_scores.mean(dim=0, keepdim=True)
            masked_scores = masked_scores.squeeze()

            # 하이퍼파라미터
            alpha = 1.0  # 평균 항 가중치
            beta = 2.0   # Log-Cosh 손실 가중치
            gamma = 2.0  # 제약 조건 페널티 가중치

            # Log-Cosh 손실 함수 정의
            def log_cosh(x):
                return torch.log(torch.cosh(x))
            
            def total_loss2(f, g, a):
                # 차이 계산
                diff = f - a * g

                # Log-Cosh 손실 (두 번째 항)
                log_cosh_term = beta * torch.mean(log_cosh(diff))
                
                return log_cosh_term

            import torch.nn as nn
            import torch.optim as optim

            b = torch.tensor(1.0, requires_grad=True)  # 스케일 파라미터
            # 옵티마이저 설정
            optimizer = optim.Adam([b], lr=0.02)

            # 학습 루프
            num_epochs = 1000
            for epoch in range(num_epochs):
                optimizer.zero_grad()  # 기울기 초기화
                loss = total_loss2(scores_1d, masked_scores, b)  # 손실 계산
                loss.backward()  # 기울기 계산
                optimizer.step()  # 최적화 진행
            
            fig, ax1 = plt.subplots(figsize=(10, 6))

            scores_1d = scores_1d.cpu().detach().numpy()
            scores_1d = (scores_1d - scores_1d.min()) / (scores_1d.max() - scores_1d.min())

            masked_scores_2 = b * masked_scores
            masked_scores_2 = masked_scores_2.cpu().detach().numpy()
            masked_scores_2 = (masked_scores_2 - masked_scores_2.min()) / (masked_scores_2.max() - masked_scores_2.min())

            num_frames = video_feature.shape[0]
            start = gt[0] / duration * num_frames
            end = gt[1] / duration * num_frames

            # 동일한 축에 두 데이터 플롯
            ax1.set_xlabel("Frame")
            ax1.set_ylabel("Similarity", color='black')
            ax1.plot(masked_scores_2, label="S_masked", linestyle='-', color='red')
            ax1.plot(scores_1d - masked_scores_2, label="S_diff", linestyle='-', color='blue')
            ax1.plot(scores_1d, label="S_original", linestyle='-', color='black')
            ax1.axvspan(start, end, color='grey', alpha=0.5, label='Timestamp')
            # 범례 추가
            ax1.legend(loc="upper left")

            # 제목 및 저장
            plt.title("Comparison of S_original and S_masked_adjusted")
            plt.grid(True)
            save_path = os.path.join(f"./plot_test2/sentence_{ann['sentences'][i]}.png")
            plt.savefig(save_path)
            plt.close()
            import pdb;pdb.set_trace()



if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
        
    with open(args.llm_output) as f:
        data = json.load(f)
    eval_test2(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'])