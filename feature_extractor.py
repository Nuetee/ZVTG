import cv2
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CLIP 모델 및 프로세서 불러오기
# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
model.to(device)

data_path = './datasets/Charades_videos/'
save_path = './dataset/charades-sta-blip/'
os.makedirs(save_path, exist_ok=True)
file_list = os.listdir(data_path)

target_fps = 3
for file in tqdm(file_list, desc="Processing video files", unit="file"):
    video_path = os.path.join(data_path, file)
    video_feature_save_path = os.path.join(save_path, file.replace(".mp4", ".pth"))
    # numpy_path = os.path.join("../../data/ActivityNet", file.replace(".mp4", ".npy"))

    cap = cv2.VideoCapture(video_path)

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_interval = source_fps / target_fps
    print(f"File name: {file} || FPS: {source_fps} || Total frames: {total_frames} || fps interval: {fps_interval}")

    frame_idx = 0
    next_frame = torch.tensor(0.)
    total_image_features = torch.tensor([]).cuda()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == int(torch.round(next_frame)):
            next_frame += fps_interval
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            with torch.no_grad():
                image = vis_processors["eval"](pil_image).unsqueeze(0).to(device)
                sample = {"image": image}
                image_features = model.extract_features(sample, mode="image")[1]
                total_image_features = torch.cat((total_image_features, image_features), dim=0)
        frame_idx += 1
    print(f"File name {file} || Frame {frame_idx} || feature shape: {total_image_features.shape}")

    # 비디오 파일 닫기
    cap.release()
    torch.save(total_image_features.cpu(), video_feature_save_path)