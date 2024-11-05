import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from llm_prompting import select_proposal

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
# model, vis_processors, text_processors = load_model_and_preprocess("clip_feature_extractor", model_type="ViT-L-14", device='cuda', is_eval=True) ### CLIP

vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])


def nms(moments, scores, pre_mom, pre_score, thresh):
    scores = scores + pre_score * 0.0
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    pre_mom = pre_mom[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i + 1:], moments[i]) > thresh
        suppressed[i + 1:][mask] = True
    return moments[~suppressed], pre_mom[~suppressed], scores[~suppressed]


def iou(candidates, gt):
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x = np.arange(-size, size + 1)
    normal = 1 / (np.sqrt(2.0 * np.pi) * sigma)
    g = np.exp(-x ** 2 / (2.0 * sigma ** 2)) * normal
    return g

def nchk(f, f1, f2, ths):
    return (((3 * f) > ths) | ((2 * f + f1) > ths) | ((f + f1 + f2) > ths))

def get_dynamic_scores(scores, stride, masks, ths=0.0005, sigma=1):
    gstride = min(stride - 2, 3)
    if (stride < 3):
        gkernel = torch.ones((1, 1, 1)).to('cuda')
    else:
        gkernel = gaussian_kernel(gstride, sigma)
        gkernel = torch.from_numpy(gkernel).float().to('cuda')
        gkernel = gkernel.view(1, 1, -1)
    gscore = F.conv1d(scores.view(-1, 1, scores.size(-1)), gkernel).view(scores.size(0), -1)

    diffres = torch.diff(gscore).to('cuda')
    pad_left = torch.zeros((diffres.size(0), (masks.size(-1) - diffres.size(-1)) // 2)).to('cuda')
    pad_right = torch.zeros((diffres.size(0), masks.size(-1) - diffres.size(-1) - pad_left.size(-1))).to('cuda')
    diffres = torch.cat((pad_left, diffres, pad_right), dim=-1) * masks

    dynamic_scores = np.zeros((diffres.size(0), diffres.size(-1)))
    dynamic_idxs = np.zeros((diffres.size(0), diffres.size(-1)))

    for idx in range(diffres.size(0)):
        f1 = f2 = f3 = 0
        d_score = 0
        d_idx = 0
        for i in range(diffres.size(-1)):
            f3 = f2
            f2 = f1
            f1 = diffres[idx][i]
            if nchk(f1, f2, f3, ths):
                d_score += max(3 * f1, 2 * f1 + f2, f1 + f2 + f3)
            else:
                d_idx = i
                d_score = 0

            dynamic_idxs[idx][i] = d_idx / scores.size(-1)
            dynamic_scores[idx][i] = d_score

    dynamic_idxs = torch.from_numpy(dynamic_idxs).to('cuda')
    dynamic_scores = torch.from_numpy(dynamic_scores).to('cuda')
    return dynamic_idxs, dynamic_scores


from sklearn.cluster import KMeans


def split_interval(init_timestep):
    init_timestep = init_timestep.cpu().sort()[0]
    # 결과를 저장할 리스트
    ranges = []

    # 임시로 시작과 끝을 저장할 변수
    start = init_timestep[0]
    end = init_timestep[0].clone()

    # 텐서의 각 원소를 순차적으로 비교
    for i in range(1, len(init_timestep)):
        if init_timestep[i] == end + 1:
            # 연속된 숫자인 경우
            end = init_timestep[i]
        else:
            # 연속되지 않은 숫자가 나타나면 구간을 추가하고 새로 시작
            ranges.append([start, end])
            start = init_timestep[i]
            end = init_timestep[i].clone()

    # 마지막 구간 추가
    ranges.append([start, end])
    return torch.tensor(ranges)

def calc_scores(video_features, sentences, gt, duration, min_cluster_sim):
    num_frames = video_features.shape[0]
    gt = torch.round(torch.tensor(gt) / torch.tensor(duration) * num_frames).to(torch.int)
    with torch.no_grad():
        # print(sentences)
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
            'cuda')
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
    v1 = F.normalize(text_feat, dim=-1)
    v2 = F.normalize(torch.tensor(video_features, device='cuda', dtype=v1.dtype), dim=-1)
    scores = torch.einsum('md,npd->mnp', v1, v2)
    scores, scores_idx = scores.max(dim=-1)
    scores = scores.mean(dim=0, keepdim=True)
    ##### scores normalization
    # scores = (scores - scores.min()) / (scores.max() - scores.min())
    ##### scores normalization

    cum_scores = torch.cumsum(scores, dim=1)[0]

    scores_idx = scores_idx.reshape(-1)
    video_features = torch.tensor(video_features).cuda()
    selected_video_features = video_features[torch.arange(num_frames), scores_idx]
    time_features = (torch.arange(num_frames) / num_frames).unsqueeze(1).cuda()
    selected_video_time_features = torch.cat((selected_video_features, time_features), dim=1)
    '''
    from sklearn.mixture import GaussianMixture
    np_scores = np.array(scores.cpu()).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(np_scores)
    prob = gmm.predict_proba(np_scores)
    prob = prob[:,gmm.means_.argmax()]
    '''

    #### feature similarity matrix 저장
    # import matplotlib.pyplot as plt
    # import os
    # normalized_features = torch.nn.functional.normalize(selected_video_time_features, p=2, dim=1)
    # cosine_similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
    # cosine_similarity_matrix_np = cosine_similarity_matrix.detach().cpu().numpy()
    # title_index = 0
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cosine_similarity_matrix_np, cmap='viridis')
    # # Timestamp 구간 표시
    # start = gt[0]
    # end = gt[1]
    # plt.axvspan(start, end, color='grey', alpha=0.5, label='Timestamp')
    # plt.colorbar()
    # plt.title("Cosine Similarity Matrix")
    # os.makedirs('./sim_matrix', exist_ok=True)
    # plt.savefig(f"./sim_matrix/cosine_similarity_matrix_{sentences}.png")
    #### feature similarity matrix 저장

    #### 비디오 프레임 벡터 스무딩 (글로벌)
    smooth_kernel_size = 21
    smooth_padding = smooth_kernel_size // 2
    padding_selected_video_time_features_global = torch.cat((selected_video_time_features[0].repeat(smooth_padding, 1),
                                                      selected_video_time_features,
                                                      selected_video_time_features[-1].repeat(smooth_padding, 1)),
                                                     dim=0)
    kernel = torch.ones(padding_selected_video_time_features_global.shape[1], 1, smooth_kernel_size).cuda() / smooth_kernel_size
    padding_selected_video_time_features_global = padding_selected_video_time_features_global.unsqueeze(0).permute(0, 2, 1)  # (1, 257, 104)
    smoothed_selected_video_time_features_global = F.conv1d(padding_selected_video_time_features_global, kernel, padding=0,
                                                     groups=padding_selected_video_time_features_global.shape[1])
    smoothed_selected_video_time_features_global = smoothed_selected_video_time_features_global.permute(0, 2, 1)
    selected_video_time_features_global = smoothed_selected_video_time_features_global[0]
    #### 비디오 프레임 벡터 스무딩 (글로벌)

    #### 비디오 프레임 벡터 스무딩 (로컬)
    smooth_kernel_size = 21
    smooth_padding = smooth_kernel_size // 2
    padding_selected_video_time_features_local = torch.cat((selected_video_time_features[0].repeat(smooth_padding, 1),
                                                      selected_video_time_features,
                                                      selected_video_time_features[-1].repeat(smooth_padding, 1)),
                                                     dim=0)
    kernel = torch.ones(padding_selected_video_time_features_local.shape[1], 1,
                        smooth_kernel_size).cuda() / smooth_kernel_size
    padding_selected_video_time_features_local = padding_selected_video_time_features_local.unsqueeze(0).permute(0, 2, 1)  # (1, 257, 104)
    smoothed_selected_video_time_features_local = F.conv1d(padding_selected_video_time_features_local, kernel, padding=0,
                                                     groups=padding_selected_video_time_features_local.shape[1])
    smoothed_selected_video_time_features_local = smoothed_selected_video_time_features_local.permute(0, 2, 1)
    selected_video_time_features_local = smoothed_selected_video_time_features_local[0]
    selected_video_time_features_local_v2 = F.normalize(selected_video_time_features_local[:,:256], dim=1)

    #### scores mean pooling
    # scores = torch.einsum('ab,cb->ac', v1, selected_video_time_features_local_v2)
    # cum_scores = torch.cumsum(scores, dim=1)[0]
    #### scores mean pooling

    # selected_video_time_features_local = selected_video_time_features #### smmoth_kernel_size = 1 하고 싶으면
    #### 비디오 프레임 벡터 스무딩 (로컬)
    '''
    top_k = 2
    merge_interval = 3 ### 3,5,7
    init_timestep = sort_index[:top_k].sort()[0]
    init_timestep_merge = init_timestep.clone()
    i = 0
    while i < len(init_timestep_merge) - 1:
        if torch.abs(init_timestep_merge[i + 1] - init_timestep_merge[i]) > 1 and torch.abs(
                init_timestep_merge[i + 1] - init_timestep_merge[i]) <= merge_interval:
            insert_tensor = torch.arange(init_timestep_merge[i] + 1, init_timestep_merge[i + 1]).cuda()
            init_timestep_merge = torch.cat((init_timestep_merge[:i + 1], insert_tensor, init_timestep_merge[i + 1:]))
        i += 1
    split_init_timestep_merge = split_interval(init_timestep_merge)
    '''

    ##### scores 에다가 kmeans 확률 곱해주기
    # kmeans_k = 4  ### 4
    # if num_frames < kmeans_k:
    #     kmeans_k = 2
    # kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=42)
    # selected_features = np.array(selected_video_time_features_global.cpu())
    # kmeans_labels_global = kmeans.fit_predict(selected_features)
    # kmeans_labels_global = torch.tensor(kmeans_labels_global)
    # distances_to_centers = kmeans.transform(selected_features)
    # probabilities = torch.softmax(torch.tensor(-distances_to_centers), dim=1)
    # probabilities = probabilities[torch.arange(len(probabilities)), kmeans_labels_global]
    # scores = scores * probabilities.cuda()
    # cum_scores = torch.cumsum(scores, dim=1)[0]
    ##### scores 에다가 kmeans 확률 곱해주기

    #### K-means 클러스터링 적용 (글로벌)
    kmeans_k = 4  ### 4
    if num_frames < kmeans_k:
        kmeans_k = 2
    kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=42)
    kmeans_labels_global = kmeans.fit_predict(np.array(selected_video_time_features_global.cpu()))
    kmeans_labels_global = torch.tensor(kmeans_labels_global)
    #### K-means 클러스터링 적용 (글로벌)
    #### sungjoon 글로벌 K-means centroids ####
    # # Get the centroids of each cluster
    # centroids = kmeans.cluster_centers_
    # local_features = np.array(selected_video_time_features_local.cpu())

    # # 각 로컬 피처에 대해 가장 가까운 글로벌 중심을 찾기
    # distances = np.linalg.norm(local_features[:, np.newaxis] - centroids, axis=2)
    # kmeans_labels_local = np.argmin(distances, axis=1)
    # kmeans_labels_local = torch.tensor(kmeans_labels_local)
    #### sungjoon 글로벌 K-means centroids ####
    
    #### K-means 클러스터링 적용 (로컬)
    kmeans_k = 4  ### 4
    if num_frames < kmeans_k:
        kmeans_k = 2
    kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=42)
    kmeans_labels_local = kmeans.fit_predict(np.array(selected_video_time_features_local.cpu()))
    kmeans_labels_local = torch.tensor(kmeans_labels_local)
    #### K-means 클러스터링 적용 (로컬)

    #### (글로벌) 클러스터링 결과에 따라 묶음 만들기
    global_proposals_set = []
    global_proposals = []
    global_proposals_scores_set = []
    global_proposals_scores = []
    start_idx = 0
    current_val = kmeans_labels_global[0]
    for i in range(1, num_frames):
        if kmeans_labels_global[i] != current_val:
            features = selected_video_features[start_idx:i+1]
            # feature 사이의 pairwise 코사인 유사도 계산
            similarity_matrix = F.cosine_similarity(
                features.unsqueeze(1),  # [k, 1, 256]
                features.unsqueeze(0),  # [1, k, 256]
                dim=2
            )
            k = similarity_matrix.shape[0]
            mask = torch.ones(k, k, dtype=bool)  # 모든 값을 True로 초기화
            mask.fill_diagonal_(False)  # 대각선 값을 False로 설정

            # 대각선 값을 제외한 평균 계산
            mean_feature_similarity = similarity_matrix[mask].mean()
            if mean_feature_similarity < min_cluster_sim:
                if len(global_proposals) > 0:
                    global_proposals_set.append(global_proposals)
                    global_proposals_scores_set.append(global_proposals_scores)
                global_proposals = []
                global_proposals_scores = []
            else:
                global_proposals.append([start_idx, i])  ### start_idx 이상, i 미만 까지 같은 레이블
                score = extract_static_score(start_idx, i, cum_scores, num_frames, scores).item()
                global_proposals_scores.append(round(score, 4))

            start_idx = i
            current_val = kmeans_labels_global[i]

    features = selected_video_features[start_idx:num_frames]
    # feature 사이의 pairwise 코사인 유사도 계산
    similarity_matrix = F.cosine_similarity(
        features.unsqueeze(1),  # [k, 1, 256]
        features.unsqueeze(0),  # [1, k, 256]
        dim=2
    )
    k = similarity_matrix.shape[0]
    mask = torch.ones(k, k, dtype=bool)  # 모든 값을 True로 초기화
    mask.fill_diagonal_(False)  # 대각선 값을 False로 설정
    
    # 대각선 값을 제외한 평균 계산
    mean_feature_similarity = similarity_matrix[mask].mean()
    if mean_feature_similarity < min_cluster_sim:
        if len(global_proposals) > 0:
            global_proposals_set.append(global_proposals)
            global_proposals_scores_set.append(global_proposals_scores)
        global_proposals = []
        global_proposals_scores = []
    else:
        global_proposals.append([start_idx, i])  ### start_idx 이상, i 미만 까지 같은 레이블
        score = extract_static_score(start_idx, i, cum_scores, num_frames, scores).item()
        global_proposals_scores.append(round(score, 4))

    global_proposals.append([num_frames, num_frames])
    global_proposals_scores.append(0)
    global_proposals_set.append(global_proposals)
    global_proposals_scores_set.append(global_proposals_scores)
    ###### important debug
    # for i in range(len(global_proposals)-1):
    #     print(global_proposals[i], global_proposals_scores[i])
    # print(f'ground truth: {gt[0]} ~ {gt[1]}')
    # import pdb; pdb.set_trace()
    ###### important debug
    #### (글로벌) 클러스터링 결과에 따라 묶음 만들기



    #### (로컬) 클러스터링 결과에 따라 묶음 만들기
    local_proposals = []
    local_proposals_scores = []
    start_idx = 0
    current_val = kmeans_labels_local[0]
    for i in range(1, num_frames):
        if kmeans_labels_local[i] != current_val:
            local_proposals.append([start_idx, i])  ### start_idx 이상, i 미만 까지 같은 레이블
            score = extract_static_score(start_idx, i, cum_scores, num_frames, scores).item()
            local_proposals_scores.append(round(score, 4))
            start_idx = i
            current_val = kmeans_labels_local[i]
    local_proposals.append([start_idx, num_frames])
    score = extract_static_score(start_idx, i, cum_scores, num_frames, scores).item()
    local_proposals_scores.append(round(score, 4))
    local_proposals.append([num_frames, num_frames])
    ###### important debug
    # for i in range(len(local_proposals)):
    #     print(local_proposals[i], local_proposals_scores[i])
    # print(f'ground truth: {gt[0]} ~ {gt[1]}')
    # import pdb; pdb.set_trace()
    ###### important debug
    #### (로컬) 클러스터링 결과에 따라 묶음 만들기

    #### 잘되는놈 공사 이전
    '''
    ### (글로벌) Extracting Global Proposals (Cartesian product)
    final_proposals = []
    final_proposals_scores = []
    for i in range(len(global_proposals)):
        for j in range(i + 1, len(global_proposals)):
            start = global_proposals[i][0]
            last = global_proposals[j][0]
            # if (last - start) > num_frames * 0.5:
            #     continue
            score = extract_static_score(start, last, cum_scores, num_frames, scores).item()
            final_proposals.append([start, last])
            final_proposals_scores.append(round(score, 4))

    final_proposals = torch.tensor(final_proposals)
    final_proposals_scores = torch.tensor(final_proposals_scores)
    value, index = final_proposals_scores.sort(descending=True)
    final_proposals = final_proposals[index]
    final_proposals_scores = final_proposals_scores[index]
    ##### important debug
    # for i in range(len(final_proposals)):
    #     print(final_proposals[i], final_proposals_scores[i])
    # print(f'ground truth: {gt[0]} ~ {gt[1]}')
    #
    # for i in range(len(local_proposals)):
    #     print(local_proposals[i], local_proposals_scores[i])
    # print(f'ground truth: {gt[0]} ~ {gt[1]}')
    # import pdb; pdb.set_trace()
    ###### important debug
    ### (글로벌) Extracting Global Proposals (Cartesian product)
    '''
    #### 잘되는놈 공사 이전
    
    ### (글로벌) Extracting Global Proposals (Cartesian product)
    final_proposals = []
    final_proposals_scores_static = []
    final_proposals_scores_avg = []
    for global_proposals in global_proposals_set:
        for i in range(len(global_proposals)):
            for j in range(i + 1, len(global_proposals)):
                start = global_proposals[i][0]
                last = global_proposals[j][0]
                # if (last - start) > num_frames * 0.5:
                #     continue
                score_static = extract_static_score(start, last, cum_scores, num_frames, scores).item()
                score_avg = extract_avg_score(start, last, cum_scores, num_frames, scores).item()

                final_proposals.append([start, last])
                final_proposals_scores_static.append(round(score_static, 4))
                final_proposals_scores_avg.append(round(score_avg, 4))

    final_proposals = torch.tensor(final_proposals)
    final_proposals_scores_static = torch.tensor(final_proposals_scores_static)
    value_static, index_static = final_proposals_scores_static.sort(descending=True)
    final_proposals_static = final_proposals[index_static]
    final_proposals_scores_static = final_proposals_scores_static[index_static]

    final_proposals_scores_avg = torch.tensor(final_proposals_scores_avg)
    value_avg, index_avg = final_proposals_scores_avg.sort(descending=True)
    final_proposals_avg = final_proposals[index_avg]
    final_proposals_scores_avg = final_proposals_scores_avg[index_avg]

    ##### important debug
    # for i in range(len(final_proposals)):
    #     print(final_proposals[i], final_proposals_scores[i])
    # print(f'ground truth: {gt[0]} ~ {gt[1]}')
    #
    # for i in range(len(local_proposals)):
    #     print(local_proposals[i], local_proposals_scores[i])
    # print(f'ground truth: {gt[0]} ~ {gt[1]}')
    # import pdb; pdb.set_trace()
    ###### important debug
    ### (글로벌) Extracting Global Proposals (Cartesian product)

    return scores, final_proposals_static[:5], final_proposals_scores_static[:5], local_proposals, local_proposals_scores
    # return scores, final_proposals[:5], final_proposals_scores[:5], local_proposals, local_proposals_scores ############### 잘되는놈 공사 이전


###################### 다른 proposals를 제거한 outer_avg 구하기
def extract_static_score_reject_other_proposal(inner_mask, outer_mask, scores):
    if outer_mask.sum() != 0:
        final_score = (inner_mask * scores).sum() / inner_mask.sum() - (outer_mask * scores).sum() / outer_mask.sum()
    else:
        final_score = (inner_mask * scores).sum() / inner_mask.sum()
    return final_score
###################### 다른 proposals를 제거한 outer_avg 구하기


###################### 중심부로부터 멀어질수록 weight를 적게 주는 gaussian 추가하기
def extract_static_gaussian_score(start, end, cum_scores, num_frames, scores):
    kernel_size = end - start
    if start == 0:
        inner_sum = cum_scores[end - 1]
        # inner_sum = scores[:end-1].sum()
    else:
        inner_sum = cum_scores[end - 1] - cum_scores[start - 1]
        # inner_sum = scores[start:end-1].sum()
    outer_sum = cum_scores[num_frames - 1] - inner_sum

    if kernel_size != num_frames:
        static_score = inner_sum / kernel_size - outer_sum / (num_frames - kernel_size)
    else:
        static_score = inner_sum / kernel_size - (scores[0][0] + scores[0][-1]) / 2  #### 임시방편 느낌
        # static_score = inner_sum / kernel_size
    return static_score
###################### 중심부로부터 멀어질수록 weight를 적게 주는 gaussian 추가하기


def extract_static_score(start, end, cum_scores, num_frames, scores):
    kernel_size = end - start
    if start == 0:
        inner_sum = cum_scores[end - 1]
    else:
        inner_sum = cum_scores[end - 1] - cum_scores[start - 1]

    outer_sum = cum_scores[num_frames - 1] - inner_sum

    if kernel_size != num_frames:
        static_score = inner_sum / kernel_size - outer_sum / (num_frames - kernel_size)
    else:
        static_score = inner_sum / kernel_size - (scores[0][0] + scores[0][-1] / 2)  #### 임시방편 느낌
        # static_score = inner_sum / kernel_size
    return static_score


def extract_avg_score(start, end, cum_scores, num_frames, scores):
    kernel_size = end - start
    if start == 0:
        inner_sum = cum_scores[end - 1]
    else:
        inner_sum = cum_scores[end - 1] - cum_scores[start - 1]
    avg_score = inner_sum / kernel_size
    return avg_score


def generate_proposal(video_features, sentences, gt, duration, stride, max_stride, min_cluster_sim, nms_thresh=0.3):
    num_frames = video_features.shape[0]
    ground_truth = [round(gt[0] / duration * num_frames, 0), round(gt[1] / duration * num_frames, 0)]
    scores, final_proposals, final_proposals_scores, local_proposals, local_proposals_scores = calc_scores(video_features, sentences, gt, duration, min_cluster_sim)
    cum_scores = torch.cumsum(scores, dim=1)[0]

    masks = (scores > 0.2).float()
    scores = scores * masks
    stride = min(stride, scores.size(-1) // 2)
    # dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)
    if len(final_proposals) > 0:
        final_proposals = final_proposals.clone()
        final_scores = final_proposals_scores.clone()
        final_prefix = final_proposals[:, 0].clone().detach()
        final_scores, sort_idx = final_scores.sort(descending=True)
        final_proposals = final_proposals[sort_idx]
        final_prefix = final_prefix[sort_idx]
    else:
        final_scores = final_proposals_scores
        final_prefix = final_proposals.clone().detach()
    ####### important debug
    # print(final_proposals.cpu()*num_frames, ground_truth, final_scores, num_frames) ### frame 단위로 통일
    ####### important debug
    return [final_proposals], [final_scores], [final_prefix], scores, cum_scores, local_proposals, local_proposals_scores, num_frames

def post_processing(proposals, local_proposals, local_proposals_scores, gt, num_frames, duration, cum_scores, scores):
    ######### Refinement Global Proposals
    proposals[:,:2] = proposals[:,:2] * num_frames / duration
    post_proposals_start = []
    post_proposals_end = []
    for i in range(len(proposals)):
        post_proposals_start.append(int(proposals[i][0]))
        post_proposals_end.append(int(proposals[i][1]))
        for j in range(len(local_proposals)-1):
            if proposals[i][0] == local_proposals[j][0]:
                if proposals[i][1] < local_proposals[j][1]:
                    post_proposals_start.append(proposals[i][0])
                    post_proposals_end.append(local_proposals[i][1])
                if proposals[i][1] == local_proposals[j][1]:
                    post_proposals_start.append(proposals[i][0])
                    post_proposals_end.append(proposals[i][1])
                if proposals[i][1] > local_proposals[j][1] and proposals[i][1] < local_proposals[j + 1][1]:
                    post_proposals_start.append(proposals[i][0])
                    post_proposals_end.append(local_proposals[j][1])

                    post_proposals_start.append(proposals[i][0])
                    post_proposals_end.append(local_proposals[j + 1][1])

            if proposals[i][0] > local_proposals[j][0] and proposals[i][0] < local_proposals[j+1][0]:
                for k in range(j, len(local_proposals)-1):
                    if proposals[i][1] > local_proposals[k][0] and proposals[i][1] < local_proposals[k + 1][0]:
                        post_proposals_start.append(local_proposals[k][0])
                        post_proposals_start.append(local_proposals[k][0])
                        post_proposals_start.append(local_proposals[k + 1][0])
                        post_proposals_start.append(local_proposals[k + 1][0])
                        post_proposals_end.append(local_proposals[k][0])
                        post_proposals_end.append(local_proposals[k+1][0])
                        post_proposals_end.append(local_proposals[k][0])
                        post_proposals_end.append(local_proposals[k+1][0])
                    if proposals[i][1] == local_proposals[k][0]:
                        post_proposals_start.append(local_proposals[k][0])
                        post_proposals_start.append(local_proposals[k + 1][0])
                        post_proposals_end.append(local_proposals[k][0])
                        post_proposals_end.append(local_proposals[k][0])

    post_proposals_start = torch.tensor(post_proposals_start, dtype=torch.int).unsqueeze(1)
    post_proposals_end = torch.tensor(post_proposals_end, dtype=torch.int).unsqueeze(1)
    # try:
    post_proposals = torch.cat((post_proposals_start, post_proposals_end), dim=1)
    # except RuntimeError:
    #     import pdb; pdb.set_trace()
    #     pass
    ######### Refinement Global Proposals

    ######## 중복인 애들 및 이상한 애들 제거
    post_final_proposals = torch.unique(post_proposals, dim=0)
    count = 0
    len_proposals = len(post_final_proposals)
    while count < len_proposals:
        if post_final_proposals[count][0] >= post_final_proposals[count][1]:
            post_final_proposals = torch.cat((post_final_proposals[:count], post_final_proposals[count+1:]), dim=0)
            len_proposals -= 1
            count -= 1
        count += 1
    if post_final_proposals[-1][0] == num_frames:
        post_final_proposals = post_final_proposals[:-1]
    # print(post_final_proposals)
    ######## 중복인 애들 및 이상한 애들 제거


    ########## 최종적인 Proposals 고르기
    #### 공사중
    # len_post_final_proposals = post_final_proposals.shape[0]
    # outer_mask = torch.ones(num_frames).cuda()
    # for i in range(len_post_final_proposals):
    #     start = post_final_proposals[i][0]
    #     last = post_final_proposals[i][1]
    #     outer_mask[start:last] = 0
    #### 공사중

    len_post_final_proposals = post_final_proposals.shape[0]
    post_final_proposals_scores = []
    for i in range(len_post_final_proposals):
        start = post_final_proposals[i][0]
        last = post_final_proposals[i][1]
        score = extract_static_score(start, last, cum_scores, num_frames, scores).item()
        ##### 공사중
        # inner_mask = torch.zeros(num_frames).cuda()
        # inner_mask[start:last] = 1
        # score = extract_static_score_reject_other_proposal(inner_mask, outer_mask, scores).item()
        ##### 공사중
        post_final_proposals_scores.append(round(score, 4))

    post_final_proposals_scores = torch.tensor(post_final_proposals_scores)
    value, index = post_final_proposals_scores.sort(descending=True)
    post_final_proposals = post_final_proposals[index]
    post_final_proposals_scores = post_final_proposals_scores[index].unsqueeze(1)
    ############
    # print(torch.tensor(proposals)[:5,:2].to(torch.int)) ### 공사중
    # print(post_final_proposals[:5]) ### 공사중
    # print(torch.tensor(proposals)[:,:2].to(torch.int))
    # print(post_final_proposals)
    # print(post_final_proposals_scores)
    # print("gt: ", round(gt[0] * num_frames / duration, 0), round(gt[1] * num_frames / duration, 0)) ### 공사중
    ############
    ########## 최종적인 Proposals 고르기


    post_final_proposals = post_final_proposals[:5]
    post_final_proposals = post_final_proposals / num_frames * duration
    post_final_proposals_scores = post_final_proposals_scores[:5]
    if post_final_proposals_scores.min() < 0:
        post_final_proposals_scores = post_final_proposals_scores + (-post_final_proposals_scores.min() + 1e-4)
    post_final_proposals_scores = post_final_proposals_scores / post_final_proposals_scores.max()

    post_final_proposals_total = torch.cat((post_final_proposals, post_final_proposals_scores), dim=1)
    ######
    # for i in range(len(proposals)):
    #     print(proposals[i])
    #
    # for i in range(len(post_final_proposals_total)):
    #     print(post_final_proposals[i], post_final_proposals_scores[i])
    # print(f'ground truth: {gt[0]} ~ {gt[1]}')
    # import pdb; pdb.set_trace()
    ######
    # post_final_proposals_total[:,:2] = post_final_proposals_total[:,:2] / num_frames * duration
    return np.array(post_final_proposals_total)


def localize(video_feature, duration, query_json, stride, max_stride, min_cluster_sim):
    answer = []
    for query in query_json:
        # import pdb; pdb.set_trace()
        gt = query['gt']
        duration = query['duration']
        proposals, scores, pre_proposals, ori_scores, ori_cum_scores, local_proposals, local_proposals_scores, num_frames = generate_proposal(video_feature, query['descriptions'], gt, duration, stride, max_stride, min_cluster_sim)

        # print(ori_scores.mean(), ori_scores.max(), ori_scores.min(), ori_scores.std())
        #### select proposal 에서 GMM score 사용
        # from sklearn.mixture import GaussianMixture
        # np_scores = np.array(ori_scores.cpu()).reshape(-1, 1)
        # gmm = GaussianMixture(n_components=2, random_state=0)
        # gmm.fit(np_scores)
        # prob = gmm.predict_proba(np_scores)
        # prob = torch.tensor(prob[:, gmm.means_.argmax()])
        # gmm_scores = torch.tensor([])
        # for i in range(len(proposals[0])):
        #     gmm_score = prob[proposals[0][i][0]:proposals[0][i][1]].mean()
        #     gmm_scores = torch.cat((gmm_scores, gmm_score.unsqueeze(0)))
        #### select proposal 에서 GMM score 사용

        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, duration], [0.0, duration], [0.0, duration]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = proposals[0][:10]
            dynamic_pred = pre_proposals[0][:10]
            # scores = gmm_scores[:10]
            # scores = scores / scores.max()
            scores = scores[0][:10]
            scores = scores / scores.max()


            # if scores.min() < 0:
            #     scores = scores + (-scores.min() + 1e-4)
            # scores = scores / scores.max()
            # scores = scores + (1- scores.max()) #### 공사중

        query['response'] = []
        for i in range(len(static_pred)):
            query['response'].append({
                'start': float(dynamic_pred[i]),
                'static_start': float(static_pred[i][0]),
                'end': float(static_pred[i][1]),
                'confidence': float(scores[i])
            })
        answer.append(query)

    proposals = []
    for t in range(5): ##################### 건들여봐야해!!! 성준아
        proposals += [[p['response'][t]['static_start'], p['response'][t]['end'], p['response'][t]['confidence']] for p
                      in answer if len(p['response']) > t]  ### only static
    proposals = np.array(proposals)
    proposals[:,:2] = proposals[:,:2] / num_frames * duration
    post_proposals = proposals
    # print(np.array(proposals)[:3])
    post_proposals = post_processing(proposals, local_proposals, local_proposals_scores, gt, num_frames, duration, ori_cum_scores, ori_scores) ### Refinement
    # print(post_proposals[:3])
    np.set_printoptions(precision=4, suppress=True)
    # print(post_proposals)
    post_proposals = select_proposal(np.array(post_proposals))
    # print(post_proposals)
    # print(gt, duration)
    # print('===================================')
    return post_proposals

