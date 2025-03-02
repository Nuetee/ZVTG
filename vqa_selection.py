import sys
import torch
import os
import numpy as np
import json
from tqdm import tqdm
import argparse
from video_feature_extraction import frame_feature_extraction 

parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
parser.add_argument('--start_idx', default=0, type=int, help='')
parser.add_argument('--end_idx', default=4885, type=int, help='')
args = parser.parse_args()

# 현재 스크립트의 디렉토리를 기준으로 상대 경로를 절대 경로로 변환
current_dir = os.path.dirname(os.path.abspath(__file__))
t2v_metrics_path = os.path.abspath(os.path.join(current_dir, "../t2v-feature-input"))

# sys.path에 추가하여 import 가능하게 만듦
sys.path.append(t2v_metrics_path)

from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model, format_question, format_answer
from t2v_metrics.models.vqascore_models.mm_utils import t5_tokenizer_image_token

# ✅ 1️⃣ CLIPT5Model 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
# vqa_model = CLIPT5Model(model_name='clip-flant5-xxl', device=device)
# tokenizer = vqa_model.tokenizer  # ✅ 모델에서 사용하는 T5Tokenizer 가져오기

# ✅ 2️⃣ 질문 & 답변 템플릿 가져오기
default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

# 이제 import 가능
import t2v_metrics
clip_flant5_score = t2v_metrics.VQAScore(model='clip-flant5-xxl') # our recommended scoring model
tokenizer = clip_flant5_score.model.tokenizer

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

def select_proposal(inputs, vid, texts, gamma=0.6):
    max_indices = inputs[:, -1]
    unique_indices = list(set(max_indices))
    num_frames = inputs[:, -2][0]

    frame_feature_list = []
    for unique_index in unique_indices:
        frame_feature = frame_feature_extraction(args.dataset, vid, unique_index, num_frames)
        frame_feature_list.append(frame_feature)

    # ✅ 4️⃣ `question_template`, `answer_template` 적용 (CLIPT5의 `forward()`와 동일)
    questions = [default_question_template.format(text) for text in texts]
    answers = [default_answer_template.format(text) for text in texts]

    # ✅ 5️⃣ `format_question()`, `format_answer()` 적용 (CLIPT5의 `forward()`와 동일)
    questions = [format_question(q, conversation_style=clip_flant5_score.model.conversational_style) for q in questions]
    answers = [format_answer(a, conversation_style=clip_flant5_score.model.conversational_style) for a in answers]

    # ✅ 6️⃣ T5 토크나이징 수행 (CLIPT5의 `forward()`와 동일)
    input_ids = [t5_tokenizer_image_token(qs, tokenizer, return_tensors="pt") for qs in questions]
    labels = [t5_tokenizer_image_token(ans, tokenizer, return_tensors="pt") for ans in answers]

    # ✅ 7️⃣ 패딩을 고려하여 시퀀스 길이를 맞춤 (CLIPT5의 `forward()`와 동일)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # `IGNORE_INDEX` 사용

    input_id_list = []
    label_list = []
    N = input_ids.shape[0]
    
    for j in range(N):
        input_id = input_ids[j].unsqueeze(0)
        input_id_list.append(input_id)
        label = labels[j].unsqueeze(0)
        label_list.append(label)

    scores = clip_flant5_score(image_features=frame_feature_list, input_ids=input_id_list, labels=label_list) # scores[i][j] is the score between image i and text j
    
    # `unique_indices`의 각 값이 `scores`에서 몇 번째 index인지 찾음
    index_mapping = {unique_idx: i for i, unique_idx in enumerate(unique_indices)}

    # `max_indices`에 맞게 score 매핑
    mapped_scores = torch.zeros(len(max_indices), scores.shape[1])

    for i, max_idx in enumerate(max_indices):
        mapped_scores[i] = scores[index_mapping[max_idx]]

    mapped_scores = mapped_scores.squeeze()
    proposals = inputs[:, :-1]
    scores = np.zeros_like(mapped_scores)

    for j in range(scores.shape[0]):
        iou = calc_iou(proposals, proposals[j])
        scores[j] += (iou ** gamma * np.array(mapped_scores)).sum()       

    idx = np.argsort(-scores)
    return inputs[idx]

with open('llm_outputs_proposals.json') as f:
    data = json.load(f)
    
# pbar = tqdm(data.items())
selected_data = list(data.items())[args.start_idx:args.end_idx]  # Slice the dictionary items
pbar = tqdm(selected_data)
ious = []
thresh = np.array([0.3, 0.5, 0.7])
recall = np.array([0, 0, 0])
for vid, ann in pbar:
    duration = ann['duration']

    for i in range(len(ann['sentences'])):
        gt = ann['timestamps'][i]
        proposals = ann['proposals'][i]
        proposals = select_proposal(np.array(proposals), vid, [ann['sentences'][i]])
        iou_ = calc_iou(proposals[:1], gt)[0]
        ious.append(max(iou_, 0))
        recall += thresh <= iou_

    pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})