import os
from tqdm import tqdm
import torch
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import boxcox
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from llm_prompting import select_proposal

model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda',
                                                                   is_eval=True)
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

import re
def sanitize_filename(filename):
    # 허용되지 않는 문자를 `_`로 대체
    filename = re.sub(r'[\/:*?"<>|]', '_', filename)
    return filename

# calc_scores 함수 실행 후에 scores와 normalized_scores를 입력으로 사용합니다.
import matplotlib.pyplot as plt
def plot_scores(scores, normalized_scores, timestamps, filename="scores_plot.png"):
    # scores와 normalized_scores를 GPU에서 CPU로 이동시키고 numpy 배열로 변환
    scores_np = scores.squeeze().cpu().numpy()
    normalized_scores_np = normalized_scores.squeeze().cpu().numpy()

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(scores_np, label="Scores", linestyle="-")
    plt.plot(normalized_scores_np, label="Normalized Scores", linestyle="-")
    plt.xlabel("Frame Index")
    plt.ylabel("Score")
    plt.title("Scores and Normalized Scores")
    plt.legend()
    plt.grid(True)
    
    # timestamps 구간을 회색으로 칠하기
    start, end = timestamps 
    plt.axvspan(start, end, color='gray', alpha=0.3)  # 회색 구간 추가

    # 그래프를 파일로 저장
    plt.savefig(filename)
    plt.close()  # 메모리 절약을 위해 그래프 닫기

def calc_scores(video_features, sentences, gt, duration, gamma, kmeans_k, prior=1):
    num_frames = video_features.shape[0]
    # gt = torch.round(torch.tensor(gt) / torch.tensor(duration) * num_frames).to(torch.int)
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

    # scores > 0.2인 마스킹 생성 (Boolean 형태 유지)
    initial_mask = (scores > 0.2)  # 0.2 이하 값은 False, 나머지는 True

    # scores의 길이가 3 미만인 경우 예외 처리
    if scores.shape[1] < 3:
        masks = initial_mask.squeeze()  # initial_mask를 그대로 사용
    else:
        # 양쪽 끝에 2씩 패딩 (모두 False로 설정)
        padded_mask = F.pad(initial_mask, (1, 1), mode='constant', value=False)

        # 현재 위치를 기준으로 양옆 2개의 마스크 값 확인
        final_mask = padded_mask.clone()  # 최종 마스크 결과 저장
        for i in range(2, padded_mask.shape[1] - 1):
            window = padded_mask[:, i - 1 : i + 2]
            if window.sum() < 2:  # 과반 이상이 마스킹되지 않은 경우
                final_mask[:, i] = 0

        # 패딩 제거하여 원래 크기의 마스크로 복원
        masks = final_mask[:, 1:-1].squeeze()
    
    # 모든 값이 False일 경우 전부 True로 설정
    if not masks.any():
        masks[:] = True

    # final_mask를 기반으로 masked_indices 계산
    masked_indices = torch.nonzero(masks, as_tuple=True)[0]  # 마스킹된 실제 인덱스 저장
    
    #### w/o sim score norm ####
    # masked_scores = scores[:, masks]
    # cum_scores = torch.cumsum(masked_scores, dim=1)[0]
    #### w/o sim score norm ####
    
    #### Similarity score noramlization ####
    device = scores.device
    data = scores[:, masks].flatten().cpu().numpy()   # 마스크된 부분만 가져오기
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
    result = minimize_scalar(neg_log_likelihood, bounds=(-2, 2), method='bounded')
    best_lambda = result.x
    
    # 최적의 lambda로 변환 데이터 생성
    transformed_data = boxcox_transformed(data, best_lambda)

    original_min, original_max = data.min(), data.max()
    transformed_min, transformed_max = transformed_data.min(), transformed_data.max()
    transformed_data = (transformed_data - transformed_min) / (transformed_max - transformed_min)  # normalize to [0, 1]
    is_scale = False
    if original_max - original_min > gamma:
        is_scale = True
        transformed_data = transformed_data * (original_max - original_min) + original_min  # scale to original min/max
    else:
        transformed_data = transformed_data * (gamma) + original_min
    # 변환 결과를 다시 텐서로 변환하고 원래 형태로 복원

    normalized_scores = torch.tensor(transformed_data, device=device).unsqueeze(0)

    cum_scores = torch.cumsum(normalized_scores, dim=1)[0]
    #### Similarity score noramlization ####
    
    #### save sim score ####
    # revise_sentences = sanitize_filename(sentences)
    # gt_frame = [int(round(x / duration * num_frames)) for x in gt]
    # plot_scores(scores, normalized_scores, gt_frame, filename=f"./sim_score/{revise_sentences}.png")
    #### save sim score ####

    scores_idx = scores_idx.reshape(-1)
    video_features = torch.tensor(video_features).cuda()
    selected_video_features = video_features[torch.arange(num_frames), scores_idx]
    time_features = (torch.arange(num_frames) / num_frames).unsqueeze(1).cuda()
    selected_video_time_features = torch.cat((selected_video_features, time_features), dim=1)
    selected_video_time_features = selected_video_time_features[masks]

    ### feature t-SNE 저장
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Normalize features
    normalized_features = torch.nn.functional.normalize(selected_video_time_features, p=2, dim=1)
    normalized_features_np = normalized_features.detach().cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(normalized_features_np)

    # Plot t-SNE with indices, using red color for points within the timestamp range
    start = gt[0]
    end = gt[1]
    plt.figure(figsize=(8, 8))
    for i, (x, y) in enumerate(tsne_features):
        color = 'red' if start <= i <= end else 'blue'
        plt.scatter(x, y, c=color, s=10)
        plt.text(x, y - 0.1, str(i), fontsize=8, ha='center')  # Show index slightly below the point

    plt.title("t-SNE of Single-Frame Features with Indices")
    plt.legend()
    # os.makedirs('./tsne', exist_ok=True)
    # plt.savefig(f"./tsne/tsne_features_{sentences}.png")
    os.makedirs('./tsne_activitynet', exist_ok=True)
    plt.savefig(f"./tsne_activitynet/tsne_features_{sentences}.png")
    ### feature t-SNE 저장

    #### 비디오 프레임 벡터 스무딩 (글로벌)
    smooth_kernel_size = 21
    smooth_padding = smooth_kernel_size // 2
    padding_selected_video_time_features_global = torch.cat((selected_video_time_features[0].repeat(smooth_padding, 1), selected_video_time_features, selected_video_time_features[-1].repeat(smooth_padding, 1)), dim=0)
    kernel = torch.ones(padding_selected_video_time_features_global.shape[1], 1, smooth_kernel_size).cuda() / smooth_kernel_size
    padding_selected_video_time_features_global = padding_selected_video_time_features_global.unsqueeze(0).permute(0, 2, 1)  # (1, 257, 104)

    padding_selected_video_time_features_global = padding_selected_video_time_features_global.float()
    smoothed_selected_video_time_features_global = F.conv1d(padding_selected_video_time_features_global, kernel, padding=0, groups=padding_selected_video_time_features_global.shape[1])
    smoothed_selected_video_time_features_global = smoothed_selected_video_time_features_global.permute(0, 2, 1)
    selected_video_time_features_global = smoothed_selected_video_time_features_global[0]
    #### 비디오 프레임 벡터 스무딩 (글로벌)

    ### feature t-SNE 저장
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Normalize features
    normalized_features = torch.nn.functional.normalize(selected_video_time_features_global, p=2, dim=1)
    normalized_features_np = normalized_features.detach().cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(normalized_features_np)

    # Plot t-SNE with indices, using red color for points within the timestamp range
    start = gt[0]
    end = gt[1]
    plt.figure(figsize=(8, 8))
    for i, (x, y) in enumerate(tsne_features):
        color = 'red' if start <= i <= end else 'blue'
        plt.scatter(x, y, c=color, s=10)
        plt.text(x, y - 0.1, str(i), fontsize=8, ha='center')  # Show index slightly below the point

    plt.title("t-SNE of Global Features with Indices")
    plt.legend()
    # os.makedirs('./tsne_global', exist_ok=True)
    # plt.savefig(f"./tsne_global/tsne_global_features_{sentences}.png")
    os.makedirs('./tsne_global_activitynet', exist_ok=True)
    plt.savefig(f"./tsne_global_activitynet/tsne_global_features_{sentences}.png")
    ### feature t-SNE 저장

    #### K-means 클러스터링 적용 (글로벌)
    if len(masked_indices) < kmeans_k:
        kmeans_k = 2
    kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=42)
    kmeans_labels_global = kmeans.fit_predict(np.array(selected_video_time_features_global.cpu()))
    kmeans_labels_global = torch.tensor(kmeans_labels_global)
    #### K-means 클러스터링 적용 (글로벌)

    #### (글로벌) 클러스터링 결과에 따라 묶음 만들기
    global_proposals = []
    global_proposals_scores = []
    start_idx = 0
    # start_idx = masked_indices[0].item()  # 마스킹된 첫 번째 인덱스
    current_val = kmeans_labels_global[0]
    for i in range(1, len(kmeans_labels_global)):
        if kmeans_labels_global[i] != current_val:
            global_proposals.append([start_idx, i])  ### start_idx 이상, i 미만 까지 같은 레이블
            score = extract_static_score(start_idx, i, cum_scores, len(cum_scores), scores).item()
            global_proposals_scores.append(round(score, 4))
            start_idx = i
            current_val = kmeans_labels_global[i]
    
    global_proposals.append([start_idx, len(kmeans_labels_global)])
    score = extract_static_score(start_idx, len(kmeans_labels_global), cum_scores, len(cum_scores), scores).item()
    global_proposals_scores.append(round(score, 4))
    global_proposals.append([len(kmeans_labels_global), len(kmeans_labels_global)])
    #### (글로벌) 클러스터링 결과에 따라 묶음 만들기

    ### (글로벌) Extracting Global Proposals (Cartesian product)
    final_proposals = []
    final_proposals_scores_static = []
    final_proposals_scores_avg = []
    for i in range(len(global_proposals)):
        for j in range(i + 1, len(global_proposals)):
            start = global_proposals[i][0]
            last = global_proposals[j][0]
            if (last - start) > num_frames * prior:
                continue
            score_static = extract_static_score(start, last, cum_scores, len(cum_scores), scores).item()
            score_avg = extract_avg_score(start, last, cum_scores, len(cum_scores), scores).item()
            
            final_proposals.append([start, last])
            final_proposals_scores_static.append(round(score_static, 4))
            final_proposals_scores_avg.append(round(score_avg, 4))

    final_proposals = [
        [
            masked_indices[start].item() if start < len(masked_indices) else num_frames,
            masked_indices[last].item() if last < len(masked_indices) else num_frames
        ]
        for start, last in final_proposals
    ]
    final_proposals = torch.tensor(final_proposals)
    final_proposals_scores_static = torch.tensor(final_proposals_scores_static)

    value_static, index_static = final_proposals_scores_static.sort(descending=True)
    final_proposals_static = final_proposals[index_static]
    final_proposals_scores_static = final_proposals_scores_static[index_static]

    final_proposals_scores_avg = torch.tensor(final_proposals_scores_avg)
    value_avg, index_avg = final_proposals_scores_avg.sort(descending=True)
    final_proposals_avg = final_proposals[index_avg]
    final_proposals_scores_avg = final_proposals_scores_avg[index_avg]
    ### (글로벌) Extracting Global Proposals (Cartesian product)

    return scores, final_proposals_static, final_proposals_scores_static

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


def generate_proposal(video_features, sentences, gt, duration, stride, max_stride, gamma, kmeans_k, prior, nms_thresh=0.3):
    num_frames = video_features.shape[0]
    ground_truth = [round(gt[0] / duration * num_frames, 0), round(gt[1] / duration * num_frames, 0)]
    scores, final_proposals, final_proposals_scores = calc_scores(video_features, sentences, gt, duration, gamma, kmeans_k, prior)
    cum_scores = torch.cumsum(scores, dim=1)[0]

    masks = (scores > 0.2).float()
    scores = scores * masks
    stride = min(stride, scores.size(-1) // 2)

    #### dynamic scoring #####
    dynamic_idxs, dynamic_scores = get_dynamic_scores(scores, stride, masks)
    dynamic_frames = torch.round(dynamic_idxs * num_frames).int()
    
    for final_proposal in final_proposals:
        current_frame = final_proposal[0]
        dynamic_prefix = dynamic_frames[0][current_frame]
        while True:
            if current_frame == 0 or dynamic_frames[0][current_frame - 1] != dynamic_prefix:
                break
            current_frame -= 1
        final_proposal[0] = current_frame
    #### dynamic scoring #####

    final_proposals = final_proposals.clone()
    final_scores = final_proposals_scores.clone()
    final_prefix = final_proposals[:, 0].clone().detach()
    final_scores, sort_idx = final_scores.sort(descending=True)
    final_proposals = final_proposals[sort_idx]
    final_prefix = final_prefix[sort_idx]
    return [final_proposals], [final_scores], [final_prefix], scores, cum_scores, num_frames


def localize(video_feature, duration, query_json, stride, max_stride, gamma, cand_num, kmeans_k, prior):
    answer = []
    for query in query_json:
        # import pdb; pdb.set_trace()
        gt = query['gt']
        proposals, scores, pre_proposals, ori_scores, ori_cum_scores, num_frames = generate_proposal(video_feature, query['descriptions'], gt, duration, stride, max_stride, gamma, kmeans_k, prior)
        if len(proposals[0]) == 0:
            static_pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
            dynamic_pred = np.array([0.0, 0.0, 0.0])
            scores = np.array([1.0, 1.0, 1.0])
        else:
            static_pred = proposals[0][:cand_num]
            dynamic_pred = pre_proposals[0][:cand_num]
            # scores = gmm_scores[:10]
            # scores = scores / scores.max()
            scores = scores[0][:cand_num]
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
    for t in range(cand_num): ##################### 건들여봐야해!!! 성준아
        proposals += [[p['response'][t]['static_start'], p['response'][t]['end'], p['response'][t]['confidence']] for p
                      in answer if len(p['response']) > t]  ### only static
    proposals = np.array(proposals)
    proposals[:,:2] = proposals[:,:2] / num_frames * duration
    post_proposals = proposals
    np.set_printoptions(precision=4, suppress=True)
    post_proposals = select_proposal(np.array(post_proposals))
    return post_proposals

