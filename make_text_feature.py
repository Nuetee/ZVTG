import json
import torch
import numpy as np
from lavis.models import load_model_and_preprocess
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

#### BLIP-2 Q-Former ####
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])
#### BLIP-2 Q-Former ####

# JSON 파일 로드
with open("dataset/activitynet/llm_outputs.json", "r") as f:
    data = json.load(f)

# 저장할 데이터 리스트
text_features_list = []
text_metadata = {}

# 데이터 전처리
for vid, details in tqdm(data.items(), desc="Processing Videos", unit="video"):
    sentences = details["sentences"]
    timestamps = details["timestamps"]  # 해당 문장의 후보 구간 정보

    for i, (sentence, timestamp) in enumerate(zip(sentences, timestamps)):
        query_id = f"{vid}_{i}"  # query ID 생성 (vid_0, vid_1, ...)
        with torch.no_grad():
            text = model.tokenizer(sentence, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
            text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
        v1 = F.normalize(text_feat, dim=-1)

        # 데이터 저장
        text_features_list.append(v1.cpu().numpy())  
        text_metadata[query_id] = {
            "text": sentence, 
            "vid": vid,
            "timestamps": timestamp  # 후보 구간 정보 추가
        }

# NumPy 배열로 변환 및 저장
text_features_array = np.vstack(text_features_list)  # (num_texts, feature_dim)
np.save("text_features.npy", text_features_array)

# JSON 파일 저장
with open("text_metadata.json", "w") as f:
    json.dump(text_metadata, f, indent=4)

print("✅ text_features.npy & text_metadata.json 생성 완료!")
