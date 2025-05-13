import json
import cv2
import torch
import numpy as np
from PIL import Image
import os
import argparse
import sys
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
t2v_metrics_path = os.path.abspath(os.path.join(current_dir, "../t2v-feature-input"))

# sys.path에 추가하여 import 가능하게 만듦
sys.path.append(t2v_metrics_path)

from custom_clip_vision_encoder import CustomCLIPVisionEncoder  # ✅ Custom Vision Tower 사용

# ✅ 모델을 전역 변수로 설정 (한 번만 로드됨)
vision_tower = None  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """✅ 모델을 한 번만 로드하는 함수"""
    print("🔹 CustomCLIPVisionEncoder 모델을 처음 로드합니다...")
    model = CustomCLIPVisionEncoder(
        model_name="openai/clip-vit-large-patch14-336", 
        select_layer=-2, 
        select_feature="patch"
    ).to(device)
    return model

def extract_frame_ffmpeg(video_path, frame_index, output_path):
    """FFmpeg를 이용해 특정 프레임을 강제로 추출"""
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select=eq(n\\,{frame_index})",
            "-vsync", "vfr",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"⚠️ FFmpeg extraction failed: {e}")
        return False

def frame_feature_extraction(dataset, vid, frame_idx, num_frames):
    global vision_tower  # ✅ 전역 변수 사용

    if vision_tower is None:
        vision_tower = load_model()  # ✅ 첫 번째 호출에서만 모델 로드됨
    # else:
    #     print("🔹 기존 모델 사용 중...")  # ✅ 반복 호출에서도 같은 모델 사용

    # ✅ Image Processor 가져오기 (전역 모델에서 가져옴)
    image_processor = vision_tower.image_processor

    # JSON 파일 불러오기
    with open(f'dataset/{dataset}/llm_outputs.json', 'r') as f:
        video_json = json.load(f)

    # 비디오 파일 경로 설정
    if dataset == 'charades-sta':
        data_path = '../PRVR/video_data/Charades_v1/'
    elif dataset == 'activitynet':
        data_path = ['../PRVR/video_data/Activitynet_1-2/', '../PRVR/video_data/Activitynet_1-3/']
    else:
        raise ValueError("Unsupported dataset. Choose from 'charades-sta' or 'activitynet'.")

    # 비디오 key로 파일 찾기
    matched_files = []
    if isinstance(data_path, list):  # ActivityNet의 경우 여러 경로에서 찾기
        for path in data_path:
            for file in os.listdir(path):
                if file.startswith(vid):
                    matched_files.append(os.path.join(path, file))
    else:  # Charades-sta의 경우 단일 경로
        for file in os.listdir(data_path):
            if file.startswith(vid):
                matched_files.append(os.path.join(data_path, file))

    if not matched_files:
        raise FileNotFoundError(f"No video file found for key: {vid}")

    video_path = matched_files[0]  # 첫 번째 매칭된 파일 사용

    # ✅ `expand2square()` 함수 (이미지 크기 맞추기)
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height: 
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    # 비디오에서 특정 프레임 가져오기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int((frame_idx / num_frames) * total_frames)
    target_frame = min(target_frame, total_frames - 1)
    if target_frame >= total_frames:
        raise ValueError(f"Requested frame index {target_frame} exceeds total frames {target_frame}.")

    # 특정 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"⚠️ OpenCV failed to read frame {target_frame}. Trying FFmpeg...")
        frame_save_path = f"temp_frame_{vid}_{target_frame}.jpg"
        if not extract_frame_ffmpeg(video_path, target_frame, frame_save_path):
            print(f"⚠️ FFmpeg extraction failed. Trying adjacent frames...")
            success = False
            for offset in range(1, 6):
                if extract_frame_ffmpeg(video_path, target_frame - offset, frame_save_path) or \
                   extract_frame_ffmpeg(video_path, target_frame + offset, frame_save_path):
                    success = True
                    break
            if not success:
                raise RuntimeError(f"❌ Failed to extract frame {target_frame} using both OpenCV and FFmpeg")
        frame = cv2.imread(frame_save_path)
        os.remove(frame_save_path)

    # 프레임을 PIL 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ✅ CustomCLIPVisionEncoder의 image_processor로 이미지 전처리
    background_color = [int(c * 255) for c in image_processor.image_mean]  
    pil_image = expand2square(pil_image, tuple(background_color))
    inputs = image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"].squeeze(0).to(device)

    # 피처 추출
    with torch.no_grad():
        feature = vision_tower(inputs.unsqueeze(0))  # ✅ 배치 형태로 입력

    return feature
