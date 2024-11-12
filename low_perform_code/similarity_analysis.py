import json
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil

def calc_iou(timestamps, proposal):
    # timestamps는 [start, end], proposal은 [start, end, score]
    ts_start, ts_end = timestamps
    prop_start, prop_end = proposal[0], proposal[1]
    
    # 교집합 구간 계산
    inter_start = max(ts_start, prop_start)
    inter_end = min(ts_end, prop_end)
    inter = max(0, inter_end - inter_start)  # 교집합 길이
    
    # 합집합 구간 계산
    union_start = min(ts_start, prop_start)
    union_end = max(ts_end, prop_end)
    union = union_end - union_start  # 합집합 길이
    
    # IoU 계산
    iou = inter / union if union > 0 else 0
    return iou

def calc_mean_cv(samples):
    mean_list = []
    cv_list = []
    for video_id, sentence_index in samples:
      sentence_data = similarities[video_id][sentence_index]
      similarity_scores = sentence_data["similarity_scores"]
      
      sample_mean = []
      sample_cv = []
      elements = ["sentence", "subject", "verb", "object", "prepositional phrase"]
      for elem in elements:
          if elem in similarity_scores:
              elem_mean = sum(similarity_scores[elem]) / len(similarity_scores[elem])
              elem_cv = np.std(similarity_scores[elem]) / elem_mean
              sample_mean.append({elem: elem_mean})
              sample_cv.append({elem: elem_cv})
      
      mean_list.append(sample_mean)
      cv_list.append(sample_cv)
      # 각 요소별 전체 평균을 계산하기 위한 딕셔너리 초기화
    element_sums = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}
    element_counts = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}

    # mean_list에서 각 요소의 값을 추출하고 합계 및 카운트를 계산
    for sample in mean_list:
        for elem_dict in sample:
            for elem, mean_value in elem_dict.items():
                element_sums[elem] += mean_value
                element_counts[elem] += 1

    # 각 요소별 전체 평균 계산
    element_mean_averages = {elem: element_sums[elem] / element_counts[elem] if element_counts[elem] > 0 else 0 for elem in element_sums}
    
    element_sums = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}
    element_counts = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}

    # mean_list에서 각 요소의 값을 추출하고 합계 및 카운트를 계산
    for sample in cv_list:
        for elem_dict in sample:
            for elem, cv_value in elem_dict.items():
                element_sums[elem] += cv_value
                element_counts[elem] += 1

    # 각 요소별 전체 평균 계산
    element_cv_averages = {elem: element_sums[elem] / element_counts[elem] if element_counts[elem] > 0 else 0 for elem in element_sums}
    
    return element_mean_averages, element_cv_averages

def calc_elements_mean(elements):
    element_sums = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}
    element_counts = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}

    # mean_list에서 각 요소의 값을 추출하고 합계 및 카운트를 계산
    for sample in elements:
        for elem_dict in sample:
            for elem, value in elem_dict.items():
                element_sums[elem] += value
                element_counts[elem] += 1

    # 각 요소별 전체 평균 계산
    element_averages = {elem: element_sums[elem] / element_counts[elem] if element_counts[elem] > 0 else 0 for elem in element_sums}
    return element_averages

def calc_elements_std(elements):
    element_sums = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}
    element_counts = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}

    # 각 요소별 값을 합계 및 카운트로 계산하여 평균을 구함
    for sample in elements:
        for elem_dict in sample:
            for elem, value in elem_dict.items():
                element_sums[elem] += value
                element_counts[elem] += 1

    # 각 요소별 평균 계산
    element_averages = {elem: element_sums[elem] / element_counts[elem] if element_counts[elem] > 0 else 0 for elem in element_sums}

    # 각 요소별로 편차 제곱을 다시 계산하기 위한 합계 초기화
    element_variance_sums = {elem: 0 for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}

    # 편차 제곱 합을 계산
    for sample in elements:
        for elem_dict in sample:
            for elem, value in elem_dict.items():
                element_variance_sums[elem] += (value - element_averages[elem]) ** 2

    # 분산(Variance)에서 표준편차(Standard Deviation)로 변환
    element_stddev = {
        elem: math.sqrt(element_variance_sums[elem] / element_counts[elem]) if element_counts[elem] > 0 else 0
        for elem in element_variance_sums
    }

    return element_stddev

def calc_diff_prop(samples):
    proposal_mean_list = []
    gt_mean_list = []
    mean_list = []
    proposal_static_score_mean_list = []
    gt_static_score_mean_list = []
    proposal_outer_ratio_mean_list = []
    gt_outer_ratio_mean_list = []

    for video_id, sentence_index in samples:
        sentence_data = similarities[video_id][sentence_index]
        similarity_scores = sentence_data["similarity_scores"]
        # 문장과 관련된 메타데이터
        timestamps = sentence_data["timestamps"]
        proposal = sentence_data["proposal"][0]
        duration = sentence_data["duration"]

        props_sample_mean = []
        props_static_score_sample_mean = []
        props_outer_ratio_sample_mean = []
        gt_sample_mean = []
        gt_static_score_sample_mean = []
        gt_outer_ratio_sample_mean = []
        sample_mean = []

        timestamps_scaled = [val / duration for val in timestamps]
        proposal_scaled = [val / duration for val in proposal]

        elements = ["sentence", "subject", "verb", "object", "prepositional phrase"]
        for elem in elements:
            if elem in similarity_scores:
                prop_start_idx = int(len(similarity_scores[elem]) * proposal_scaled[0])
                prop_end_idx = min(int(len(similarity_scores[elem]) * proposal_scaled[1]), len(similarity_scores[elem]) - 1)
                gt_start_idx = int(len(similarity_scores[elem]) * timestamps_scaled[0])
                gt_end_idx = min(int(len(similarity_scores[elem]) * timestamps_scaled[1]), len(similarity_scores[elem]) - 1)

                prop_mean = sum(similarity_scores[elem][prop_start_idx:prop_end_idx]) / (prop_end_idx - prop_start_idx + 1)
                gt_mean = sum(similarity_scores[elem][gt_start_idx:gt_end_idx]) / (gt_end_idx - gt_start_idx + 1)
                elem_mean = sum(similarity_scores[elem]) / len(similarity_scores[elem])
                if len(similarity_scores[elem]) - (prop_end_idx - prop_start_idx + 1) > 0:
                    prop_outer_mean = (sum(similarity_scores[elem][0:prop_start_idx]) + sum(similarity_scores[elem][prop_end_idx:len(similarity_scores[elem]) - 1])) / (len(similarity_scores[elem]) - (prop_end_idx - prop_start_idx + 1))
                    props_outer_ratio_sample_mean.append({elem: prop_mean / prop_outer_mean})
                else:
                    prop_outer_mean = 0
                
                props_static_score_sample_mean.append({elem: prop_mean - prop_outer_mean})
                
                if len(similarity_scores[elem]) - (gt_end_idx - gt_start_idx + 1) > 0:
                    gt_outer_mean = (sum(similarity_scores[elem][0:gt_start_idx]) + sum(similarity_scores[elem][gt_end_idx:len(similarity_scores[elem]) - 1])) / (len(similarity_scores[elem]) - (gt_end_idx - gt_start_idx + 1))
                    gt_outer_ratio_sample_mean.append({elem: gt_mean / gt_outer_mean})
                else:
                    gt_outer_mean = 0
                
                gt_static_score_sample_mean.append({elem: gt_mean - gt_outer_mean})

                props_sample_mean.append({elem: prop_mean})
                gt_sample_mean.append({elem: gt_mean})
                sample_mean.append({elem: elem_mean})
            
        proposal_mean_list.append(props_sample_mean)
        gt_mean_list.append(gt_sample_mean)
        mean_list.append(sample_mean)
        proposal_static_score_mean_list.append(props_static_score_sample_mean)
        gt_static_score_mean_list.append(gt_static_score_sample_mean)
        proposal_outer_ratio_mean_list.append(props_outer_ratio_sample_mean)
        gt_outer_ratio_mean_list.append(gt_outer_ratio_sample_mean)

    return proposal_mean_list, gt_mean_list, mean_list,proposal_static_score_mean_list,gt_static_score_mean_list,proposal_outer_ratio_mean_list, gt_outer_ratio_mean_list,

def calc_static_score_per_elements(samples):
    prop_static_scores_by_element1 = {elem: [] for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}
    prop_static_scores_by_element2 = {elem: [] for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}
    gt_static_scores_by_element1 = {elem: [] for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}
    gt_static_scores_by_element2 = {elem: [] for elem in ["sentence", "subject", "verb", "object", "prepositional phrase"]}
    
    for video_id, sentence_index in samples:
        sentence_data = similarities[video_id][sentence_index]
        similarity_scores = sentence_data["similarity_scores"]
        timestamps = sentence_data["timestamps"]
        proposal = sentence_data["proposal"][0]
        duration = sentence_data["duration"]

        timestamps_scaled = [val / duration for val in timestamps]
        proposal_scaled = [val / duration for val in proposal]
        
        elements = ["sentence", "subject", "verb", "object", "prepositional phrase"]

        for elem in elements:
            if elem in similarity_scores:
                prop_start_idx = int(len(similarity_scores[elem]) * proposal_scaled[0])
                prop_end_idx = min(int(len(similarity_scores[elem]) * proposal_scaled[1]), len(similarity_scores[elem]) - 1)
                gt_start_idx = int(len(similarity_scores[elem]) * timestamps_scaled[0])
                gt_end_idx = min(int(len(similarity_scores[elem]) * timestamps_scaled[1]), len(similarity_scores[elem]) - 1)

                prop_static_score1, prop_static_score2 = calc_staic_score(prop_start_idx, prop_end_idx, similarity_scores[elem])
                gt_static_score1, gt_static_score2 = calc_staic_score(gt_start_idx, gt_end_idx, similarity_scores[elem])

                prop_static_scores_by_element1[elem].append(prop_static_score1)
                prop_static_scores_by_element2[elem].append(prop_static_score2)
                gt_static_scores_by_element1[elem].append(gt_static_score1)
                gt_static_scores_by_element2[elem].append(gt_static_score2)

    return prop_static_scores_by_element1, prop_static_scores_by_element2, gt_static_scores_by_element1, gt_static_scores_by_element2

def calc_staic_score(start, end, similairty):
    prop_mean = sum(similairty[start:end]) / (end - start + 1)
    if start == 0 and end == len(similairty) - 1:
        left_sim = min(similairty[0], sum(similairty) / len(similairty))
        right_sim = min(similairty[len(similairty) - 1], sum(similairty) / len(similairty))
        outer_mean2 = (left_sim + right_sim) / 2
        outer_mean1 = 0
    elif start == 0:
        left_sim = min(similairty[0], sum(similairty) / len(similairty))
        outer_mean2 = (left_sim * (end - start + 1) + sum(similairty[end:len(similairty) - 1])) / ((end - start + 1) + (len(similairty) - end - 1))
        outer_mean1 = (sum(similairty[0:start]) + sum(similairty[end:len(similairty) - 1])) / (len(similairty) - (end - start + 1))
    elif end == len(similairty) - 1:
        right_sim = min(similairty[len(similairty) - 1], sum(similairty) / len(similairty))
        outer_mean2 = (sum(similairty[0:start]) + right_sim * (end - start + 1)) / ((start) + (end - start + 1))
        outer_mean1 = (sum(similairty[0:start]) + sum(similairty[end:len(similairty) - 1])) / (len(similairty) - (end - start + 1))
    else:
        if (len(similairty) - (end - start + 1)) == 0:
            print(start)
            print(end)
            print(len(similairty))
        outer_mean1 = (sum(similairty[0:start]) + sum(similairty[end:len(similairty) - 1])) / (len(similairty) - (end - start + 1))
        outer_mean2 = outer_mean1
    
    static_score1 = prop_mean - outer_mean1
    static_score2 = prop_mean - outer_mean2

    return static_score1, static_score2

def calc_0_ratio(scores):
    # 양수(1) 또는 음수(0)로 변환
    binary_scores = [1 if score > 0 else 0 for score in scores]
    
    # 0의 비율 계산
    count_1 = sum(binary_scores)
    count_0 = len(binary_scores) - count_1
    total_count = len(binary_scores)
    p_0 = count_0 / total_count if total_count > 0 else 1e-10

    return p_0

def calc_dataset_0_ratio(samples):
    prop_0_ratio_list1 = []
    prop_0_ratio_list2 = []
    gt_0_ratio_list1 = []
    gt_0_ratio_list2 = []
    
    for video_id, sentence_index in samples:
        sentence_data = similarities[video_id][sentence_index]
        similarity_scores = sentence_data["similarity_scores"]
        timestamps = sentence_data["timestamps"]
        proposal = sentence_data["proposal"][0]
        duration = sentence_data["duration"]

        timestamps_scaled = [val / duration for val in timestamps]
        proposal_scaled = [val / duration for val in proposal]
        
        elements = ["sentence", "subject", "verb", "object", "prepositional phrase"]
        prop_static_scores_1 = []
        prop_static_scores_2 = []
        gt_static_scores_1 = []
        gt_static_scores_2 = []

        for elem in elements:
            if elem in similarity_scores:
                prop_start_idx = int(len(similarity_scores[elem]) * proposal_scaled[0])
                prop_end_idx = min(int(len(similarity_scores[elem]) * proposal_scaled[1]), len(similarity_scores[elem]) - 1)
                gt_start_idx = int(len(similarity_scores[elem]) * timestamps_scaled[0])
                gt_end_idx = min(int(len(similarity_scores[elem]) * timestamps_scaled[1]), len(similarity_scores[elem]) - 1)

                prop_static_score1, prop_static_score2 = calc_staic_score(prop_start_idx, prop_end_idx, similarity_scores[elem])
                gt_static_score1, gt_static_score2 = calc_staic_score(gt_start_idx, gt_end_idx, similarity_scores[elem])

                prop_static_scores_1.append(prop_static_score1)
                prop_static_scores_2.append(prop_static_score2)
                gt_static_scores_1.append(gt_static_score1)
                gt_static_scores_2.append(gt_static_score2)

        prop_0_ratio_1 = calc_0_ratio(prop_static_scores_1)
        prop_0_ratio_2 = calc_0_ratio(prop_static_scores_2)
        gt_0_ratio_1 = calc_0_ratio(gt_static_scores_1)
        gt_0_ratio_2 = calc_0_ratio(gt_static_scores_2)

        prop_0_ratio_list1.append(prop_0_ratio_1)
        prop_0_ratio_list2.append(prop_0_ratio_2)
        gt_0_ratio_list1.append(gt_0_ratio_1)
        gt_0_ratio_list2.append(gt_0_ratio_2)

    def calc_mean(list):
        return sum(list) / len(list)
    
    return calc_mean(prop_0_ratio_list1), calc_mean(prop_0_ratio_list2), calc_mean(gt_0_ratio_list1), calc_mean(gt_0_ratio_list2)

def calc_entropy(scores):
    # 양수(1) 또는 음수(0)로 변환
    binary_scores = [1 if score > 0 else 0 for score in scores]
    
    # 1과 0의 비율 계산
    count_1 = sum(binary_scores)
    count_0 = len(binary_scores) - count_1
    total_count = len(binary_scores)
    
    # 각 비율 계산 (0인 경우 방지)
    p_1 = count_1 / total_count if total_count > 0 else 1e-10
    p_0 = count_0 / total_count if total_count > 0 else 1e-10
    
    # 로그 계산 시 0을 방지하기 위한 처리
    if p_1 == 0:
        entropy_p1 = 0
    else:
        entropy_p1 = p_1 * math.log2(p_1)
        
    if p_0 == 0:
        entropy_p0 = 0
    else:
        entropy_p0 = p_0 * math.log2(p_0)

    # 엔트로피 계산
    entropy = - (entropy_p1 + entropy_p0)
    return entropy

def calc_dataset_entropy(samples):
    prop_entropy_list1 = []
    prop_entropy_list2 = []
    gt_entropy_list1 = []
    gt_entropy_list2 = []
    
    for video_id, sentence_index in samples:
        sentence_data = similarities[video_id][sentence_index]
        similarity_scores = sentence_data["similarity_scores"]
        timestamps = sentence_data["timestamps"]
        proposal = sentence_data["proposal"][0]
        duration = sentence_data["duration"]

        timestamps_scaled = [val / duration for val in timestamps]
        proposal_scaled = [val / duration for val in proposal]
        
        elements = ["sentence", "subject", "verb", "object", "prepositional phrase"]
        prop_static_scores_1 = []
        prop_estatic_scores_2 = []
        gt_static_scores_1 = []
        gt_static_scores_2 = []

        for elem in elements:
            if elem in similarity_scores:
                prop_start_idx = int(len(similarity_scores[elem]) * proposal_scaled[0])
                prop_end_idx = min(int(len(similarity_scores[elem]) * proposal_scaled[1]), len(similarity_scores[elem]) - 1)
                gt_start_idx = int(len(similarity_scores[elem]) * timestamps_scaled[0])
                gt_end_idx = min(int(len(similarity_scores[elem]) * timestamps_scaled[1]), len(similarity_scores[elem]) - 1)

                prop_static_score1, prop_static_score2 = calc_staic_score(prop_start_idx, prop_end_idx, similarity_scores[elem])
                gt_static_score1, gt_static_score2 = calc_staic_score(gt_start_idx, gt_end_idx, similarity_scores[elem])

                prop_static_scores_1.append(prop_static_score1)
                prop_estatic_scores_2.append(prop_static_score2)
                gt_static_scores_1.append(gt_static_score1)
                gt_static_scores_2.append(gt_static_score2)

        prop_entropy1 = calc_entropy(prop_static_scores_1)
        prop_entropy2 = calc_entropy(prop_estatic_scores_2)
        gt_entropy1 = calc_entropy(gt_static_scores_1)
        gt_entropy2 = calc_entropy(gt_static_scores_2)

        prop_entropy_list1.append(prop_entropy1)
        prop_entropy_list2.append(prop_entropy2)
        gt_entropy_list1.append(gt_entropy1)
        gt_entropy_list2.append(gt_entropy2)

    def calc_entropy_mean(entropy_list):
        return sum(entropy_list) / len(entropy_list)
    
    return calc_entropy_mean(prop_entropy_list1), calc_entropy_mean(prop_entropy_list2), calc_entropy_mean(gt_entropy_list1), calc_entropy_mean(gt_entropy_list2)
    
def split_samples_by_fixed_size(all_samples, split_size):
    iou_list = []
    
    # 모든 샘플에 대해 iou 계산
    for video_id, sentence_index in all_samples:
        sentence_data = similarities[video_id][sentence_index]
        timestamps = sentence_data["timestamps"]
        proposal = sentence_data["proposal"][0]
        
        iou = calc_iou(timestamps, proposal)
        
        iou_list.append({
            "vid": video_id,
            "sentence_index": sentence_index,
            "iou": iou
        })
    
    # IoU 값을 기준으로 정렬
    sorted_iou_list = sorted(iou_list, key=lambda x: x['iou'], reverse=True)

    split_samples = []
    total_samples = len(sorted_iou_list)
    current_idx = 0

    # 고정된 크기만큼 구간별로 슬라이싱
    while current_idx < total_samples:
        end_idx = min(current_idx + split_size, total_samples)
        
        # IoU 값 제거하고 (video_id, sentence_index) 형태로 clean한 리스트 생성
        split_group = [(item["vid"], item["sentence_index"]) for item in sorted_iou_list[current_idx:end_idx]]
        split_samples.append(split_group)
        current_idx = end_idx
    
    return split_samples

def top_bottom_samples(all_samples, number_of_samples):
    top_ious = []
    bottom_ious = []
    for video_id, sentence_index in all_samples:
        sentence_data = similarities[video_id][sentence_index]
        # 문장과 관련된 메타데이터
        timestamps = sentence_data["timestamps"]
        proposal = sentence_data["proposal"][0]
        
        iou = calc_iou(timestamps, proposal)

        if len(top_ious) < number_of_samples or iou > min(top_ious, key=lambda x: x['iou'])['iou']:
            if len(top_ious) >= number_of_samples:
                min_item = min(top_ious, key=lambda x: x['iou'])
                top_ious.remove(min_item)

            new_item = {
                "vid": video_id,
                "sentence_index": sentence_index,
                "iou": iou
            }
            top_ious.append(new_item)

        if len(bottom_ious) < (number_of_samples + 500) or iou < max(bottom_ious, key=lambda x: x['iou'])['iou']:
            if len(bottom_ious) >= (number_of_samples + 500):
              max_item = max(bottom_ious, key=lambda x: x['iou'])
              bottom_ious.remove(max_item)

            new_item = {
                "vid": video_id,
                "sentence_index": sentence_index,
                "iou": iou
            }
            bottom_ious.append(new_item)
    
    bottom_ious = sorted(bottom_ious, key=lambda x: x['iou'])[:number_of_samples]
    # iou를 제거한 형태로 반환
    top_ious_cleaned = [(item["vid"], item["sentence_index"]) for item in top_ious]
    bottom_ious_cleaned = [(item["vid"], item["sentence_index"]) for item in bottom_ious]

    return top_ious_cleaned, bottom_ious_cleaned

# JSON 파일 로드
with open('similarities_per_frame.json', 'r', encoding='utf-8') as f:
    similarities = json.load(f)


# 랜덤으로 10개의 비디오-문장 쌍을 선택하고 그래프 그리기
all_samples = [(vid, i) for vid in similarities for i in range(len(similarities[vid]))]
split_groups = split_samples_by_fixed_size(all_samples, 50)
elements = ["sentence", "subject", "verb", "object", "prepositional phrase"]
for split in split_groups:
    _, _, _, proposal_static_score_mean_list, gt_static_score_mean_list, _, _, = calc_diff_prop(split)
    for elem in elements:
        print(elem, ': ', proposal_static_score_mean_list[elem])
    print('\n')

# print("All sample Coefficient of variation")
# for elem, avg in all_cv.items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')

# top_samples, bottom_samples = top_bottom_samples(all_samples, 100)
# top_mean, top_cv = calc_mean_cv(top_samples)
# bottom_mean, bottom_cv = calc_mean_cv(bottom_samples)
# print("Top 100 sample mean")
# for elem, avg in top_mean.items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 Coefficient of variation")
# for elem, avg in top_cv.items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 sample mean")
# for elem, avg in bottom_mean.items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 Coefficient of variation")
# for elem, avg in bottom_cv.items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')

# proposal_mean_list, gt_mean_list, mean_list, proposal_static_score_mean_list, gt_static_score_mean_list,proposal_outer_ratio_mean_list, gt_outer_ratio_mean_list, = calc_diff_prop(top_samples)

# print("Top 100 proposal mean")
# for elem, avg in calc_elements_mean(proposal_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 proposal std")
# for elem, avg in calc_elements_std(proposal_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 proposal static score mean")
# for elem, avg in calc_elements_mean(proposal_static_score_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 proposal static score std")
# for elem, avg in calc_elements_std(proposal_static_score_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 proposal/outer mean")
# for elem, avg in calc_elements_mean(proposal_outer_ratio_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 proposal/outer std")
# for elem, avg in calc_elements_std(proposal_outer_ratio_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')

# print("Top 100 GT mean")
# for elem, avg in calc_elements_mean(gt_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 GT std")
# for elem, avg in calc_elements_std(gt_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 GT static score mean")
# for elem, avg in calc_elements_mean(gt_static_score_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 GT static score std")
# for elem, avg in calc_elements_std(gt_static_score_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 GT/outer mean")
# for elem, avg in calc_elements_mean(gt_outer_ratio_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Top 100 GT/outer std")
# for elem, avg in calc_elements_std(gt_outer_ratio_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')

# proposal_mean_list, gt_mean_list, mean_list, proposal_static_score_mean_list, gt_static_score_mean_list,proposal_outer_ratio_mean_list, gt_outer_ratio_mean_list, = calc_diff_prop(bottom_samples)
# print("Bottom 100 proposal mean")
# for elem, avg in calc_elements_mean(proposal_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 proposal std")
# for elem, avg in calc_elements_std(proposal_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 proposal static score mean")
# for elem, avg in calc_elements_mean(proposal_static_score_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 proposal static score std")
# for elem, avg in calc_elements_std(proposal_static_score_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 proposal/outer mean")
# for elem, avg in calc_elements_mean(proposal_outer_ratio_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 proposal/outer std")
# for elem, avg in calc_elements_std(proposal_outer_ratio_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')

# print("Bottom 100 GT mean")
# for elem, avg in calc_elements_mean(gt_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 GT std")
# for elem, avg in calc_elements_std(gt_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 GT static score mean")
# for elem, avg in calc_elements_mean(gt_static_score_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 GT static score std")
# for elem, avg in calc_elements_std(gt_static_score_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 GT/outer mean")
# for elem, avg in calc_elements_mean(gt_outer_ratio_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')
# print("Bottom 100 GT/outer std")
# for elem, avg in calc_elements_std(gt_outer_ratio_mean_list).items():
#     print(f"{elem}: {avg:.2f}")
# print('\n')