import json
import os
import torch
import numpy as np
import torch.nn.functional as F
from twfinch_revise import FINCH
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torchvision import transforms

#### BLIP-2 Q-Former ####
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda', is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])
#### BLIP-2 Q-Former ####

with open('dataset/charades-sta/llm_outputs.json') as f:
    data = json.load(f)

feature_path =  './datasets/Charades/'
pbar = tqdm(data.items())
for vid, ann in pbar:
    duration = ann['duration']
    video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
    num_frames = video_feature.shape[0]

    for i in range(len(ann['sentences'])):
        sentences = ann['sentences'][i]
    
        with torch.no_grad():
            text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
                'cuda')
            text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
        v1 = F.normalize(text_feat, dim=-1)
        v2 = F.normalize(torch.tensor(video_feature, device='cuda', dtype=v1.dtype), dim=-1)
        scores = torch.einsum('md,npd->mnp', v1, v2)
        scores, scores_idx = scores.max(dim=-1)
        scores = scores.mean(dim=0, keepdim=True)
        scores_idx = scores_idx.reshape(-1)
        selected_video_features = video_feature[np.arange(num_frames), scores_idx.cpu().numpy()]
        
        import pdb;pdb.set_trace()
        c, num_clust, req_c = FINCH(selected_video_features, req_clust=9, tw_finch=True)
