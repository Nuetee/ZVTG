import argparse
import numpy as np
import json
import torch
import os
import torch.nn.functional as F
from data_configs import DATASETS
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torchvision import transforms
import matplotlib.pyplot as plt

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use nonly VLM for evaluation.')
    parser.add_argument('--fixed_length', action='store_true')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--ratio', type=int, default=9)
    parser.add_argument('--unit', type=float, default=1)

    return parser.parse_args()

if __name__=='__main__':
    args = get_args()

    dataset = DATASETS[args.dataset]
    feature_path = dataset['feature_path']

    with open(args.llm_output) as f:
        data = json.load(f)
    
    pbar = tqdm(data.items())
    selected_shift_indices = {i:0 for i in np.arange(-args.ratio, args.ratio + args.unit, args.unit)}
    extreme_shift_sims = {i: [] for i in [-args.ratio, args.ratio]}  # 양 끝 인덱스 유사도 저장


    for vid, ann in pbar:
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        v2 = F.normalize(torch.tensor(video_feature, device='cuda', dtype=torch.float32), dim=-1)
        duration = ann['duration']
        num_frames = video_feature.shape[0]

        for i, sentence in enumerate(ann['sentences']): 
            with torch.no_grad():
                text = model.tokenizer(sentence, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
                text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
                text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
            v1 = F.normalize(text_feat, dim=-1)
            scores = torch.einsum('md,npd->mnp', v1, v2)
            scores, scores_idx = scores.max(dim=-1)
            scores = scores.mean(dim=0, keepdim=True)

            gt = ann['timestamps'][i]
            start_frame = int((gt[0] / duration) * (num_frames - 1))
            end_frame = int((gt[1] / duration) * (num_frames - 1))
            gt_frame = [start_frame, end_frame]

            interval_length = end_frame - start_frame

            if args.fixed_length:
                ### fixed length ###
                if start_frame - (interval_length * 0.1 * args.ratio) < 0 or end_frame + (interval_length * 0.1 * args.ratio) >= num_frames :
                    continue
                ### fixed length ###

            shifted_intervals_frame = []
            shift_info = []
            unique_intervals = set()

            for shifted_ratio in np.arange(args.unit, args.ratio + args.unit, args.unit):
                shift_amount = int(interval_length * shifted_ratio * 0.1)  # 프레임 단위 shift

                # 왼쪽으로 N * 10% 이동한 구간
                new_start_left = max(0, start_frame - shift_amount)
                new_end_left = max(0, end_frame - shift_amount)

                if [new_start_left, new_end_left] != gt_frame and tuple([new_start_left, new_end_left]) not in unique_intervals:
                    if args.fixed_length:
                        if new_end_left - new_start_left == interval_length :
                            shifted_intervals_frame.append([new_start_left, new_end_left])
                            shift_info.append(shifted_ratio * (-1))
                            unique_intervals.add(tuple([new_start_left, new_end_left]))
                    else:
                        if new_start_left != new_end_left:
                            shifted_intervals_frame.append([new_start_left, new_end_left])
                            shift_info.append(shifted_ratio * (-1))
                            unique_intervals.add(tuple([new_start_left, new_end_left]))
                
                # 오른쪽으로 N * 10% 이동한 구간
                new_start_right = min(num_frames - 1, start_frame + shift_amount)
                new_end_right = min(num_frames - 1, end_frame + shift_amount)

                if [new_start_right, new_end_right] != gt_frame and tuple([new_start_right, new_end_right]) not in unique_intervals:
                    if args.fixed_length:
                        if new_end_right - new_start_right == interval_length:
                            shifted_intervals_frame.append([new_start_right, new_end_right])
                            shift_info.append(shifted_ratio)
                            unique_intervals.add(tuple([new_start_right, new_end_right]))
                    else:
                        if new_start_right != new_end_right:
                            shifted_intervals_frame.append([new_start_right, new_end_right])
                            shift_info.append(shifted_ratio)
                            unique_intervals.add(tuple([new_start_right, new_end_right]))
            # import pdb;pdb.set_trace()
            if args.static:
                # 내부 평균 유사도 계산
                gt_internal_sim = torch.mean(scores[:, gt_frame[0]:gt_frame[1]]).item()

                # 외부 평균 유사도 계산 (구간 외부 전체 평균)
                external_mask = torch.ones_like(scores)
                external_mask[:, gt_frame[0]:gt_frame[1]] = 0
                external_values = scores[external_mask.bool()].mean().item()
                
                gt_sim = gt_internal_sim - external_values
                
                if len(shifted_intervals_frame) > 0:
                    shifted_sims = []
                    for s, e in shifted_intervals_frame:
                        internal_sim = torch.mean(scores[:, s:e]).item()
                        external_mask = torch.ones_like(scores)
                        external_mask[:, s:e] = 0
                        external_values = scores[external_mask.bool()].mean().item()
                        shifted_sims.append(internal_sim - external_values)
                    
                    max_sim = torch.max(torch.tensor(shifted_sims)).item()
                    shifted_sims_max_index = torch.tensor(shifted_sims).argmax().item()
                    max_selected_shift_index = shift_info[shifted_sims_max_index]
            else:
                # GT similarity 계산
                gt_sim = torch.mean(scores[:, gt_frame[0] : gt_frame[1]]).item()
                if len(shifted_intervals_frame) > 0:
                    shifted_sims = [torch.mean(scores[:, s:e]) for s, e in shifted_intervals_frame]
                    max_sim = torch.max(torch.stack(shifted_sims)).item()
                    shifted_sims_max_index = torch.stack(shifted_sims).argmax().item()
                    max_selected_shift_index = shift_info[shifted_sims_max_index]
            
            if gt_sim > max_sim:
                selected_shift_indices[0] += 1
            else:
                selected_shift_indices[max_selected_shift_index] += 1
                if max_selected_shift_index in extreme_shift_sims:
                    extreme_shift_sims[max_selected_shift_index].append(max_sim)

    keys = [k * 10 for k in selected_shift_indices.keys()]
    values = list(selected_shift_indices.values())  # y축: 해당 키의 값 (정수 값)
    
    total = sum(values)  # 전체 값의 합
    percentages = [(v / total) * 100 for v in values]  # 백분율 변환

    # 그래프 설정
    plt.bar(keys, percentages, width=10 * args.unit, align='center', edgecolor='black')

    # 그래프 설정
    plt.xlabel("Relative Offset (%) to Ground Truth Segment")
    plt.ylabel("roportion of Selections (%)")
    plt.ylim(0, max(10, max(percentages) * 1.05))  # 최소 10% 이상 보이도록 설정
    plt.title("Segment Selection Frequency by Relative Position")
    plt.xticks(keys, rotation=45)  # x축 눈금 설정
    plt.tight_layout()

    # 그래프 표시
    plt.savefig(f"Selection Frequency_x{args.unit}_{args.dataset}{' static' if args.static else ''}{' fixed length' if args.fixed_length else ''}{' ' + str(args.ratio)}.png", dpi=300, bbox_inches='tight')

    plt.close()  # 메모리 해제

    for i in range(0, len(percentages) // 2):
        if i == 0:
            percentages_sum = sum(percentages)
        else:
            percentages_sum = sum(percentages[i:-i])
        print(f"{keys[i]}~{keys[-(i+1)]}: {percentages_sum}")