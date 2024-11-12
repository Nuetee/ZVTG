from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from torch.nn.functional import softmax
from PIL import Image

# 모델 및 프로세서 설정
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 로컬 이미지 불러오기
image_path = "./3MSZA.jpg"  # 로컬 이미지 경로
image = Image.open(image_path)

# 마스킹된 텍스트 입력 설정
masked_text = "This image depicts a scene. Fill in the word at '[MASK]' in the following sentence. Only output the missing word: A person flipped [MASK] near the door."

inputs = processor(images=image, text=masked_text, return_tensors="pt").to(device, torch.float16)

# 초기 입력 설정
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# 반복적으로 텍스트 생성하기
generated_tokens = []
max_length = 30  # 최대 생성 토큰 길이

import pdb;pdb.set_trace()
for _ in range(max_length):
    # 모델에서 로짓(logits) 계산
    outputs = model(**inputs)
    logits = outputs.logits  # [batch_size, sequence_length, vocab_size]
    
    # 마지막 위치의 로짓에서 가장 높은 확률의 토큰 선택
    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    # 선택된 토큰을 결과에 추가하고 input_ids를 업데이트
    generated_tokens.append(next_token_id.item())
    
    # next_token_id의 차원을 input_ids와 맞추기 위해 [1, 1]로 reshape
    next_token_id = next_token_id.unsqueeze(-1)  # [batch_size, 1]
    inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_id], dim=-1)

    # 선택된 토큰이 종료 토큰([EOS])이면 중지
    if next_token_id.item() == processor.tokenizer.eos_token_id:
        break
pdb.set_trace()
# 토큰들을 텍스트로 변환
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("생성된 텍스트:", generated_text)

# # 모델에서 로짓을 추출
# with torch.no_grad():
#     outputs = model(**inputs)
# import pdb;pdb.set_trace()
# # 로짓에서 토큰별 확률 계산
# logits = outputs.logits  # [batch_size, sequence_length, vocab_size] 형태
# mask_token_index = (inputs.input_ids == processor.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]  # MASK 위치 찾기
# mask_logits = logits[0, mask_token_index, :]  # MASK 위치의 로짓 값

# # 소프트맥스를 통해 확률 계산
# mask_token_probs = softmax(mask_logits, dim=-1)

# # 토큰별 확률 값을 출력 (상위 몇 개의 확률 값만 확인)
# top_k = 5
# top_k_probs, top_k_indices = torch.topk(mask_token_probs, top_k, dim=-1)
# top_k_tokens = [processor.tokenizer.decode([idx]) for idx in top_k_indices[0]]

# print("Top K 예측 단어 및 확률:")
# for token, prob in zip(top_k_tokens, top_k_probs[0]):
#     print(f"{token}: {prob.item()}")
