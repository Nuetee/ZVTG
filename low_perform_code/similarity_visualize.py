import json
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# ActivityNet은 2fps, Charades는 3fps
def plot_similarity_graph(video_id, sentence_index, similarities, save_dir, fps=2):
    # 유사도 데이터를 가져옴
    sentence_data = similarities[video_id][sentence_index]
    
    # 문장과 관련된 메타데이터
    sentence = sentence_data["sentence"]
    duration = sentence_data["duration"]
    timestamps = sentence_data["timestamps"]
    proposal = sentence_data["proposal"][0]
    similarity_scores = sentence_data["similarity_scores"]
    elements_obj = sentence_data["elements"]
    
    # 시간축 설정: 프레임 수에 따라 전체 duration을 나눔
    num_frames = len(next(iter(similarity_scores.values())))  # 첫 번째 유효한 값의 길이로 프레임 수 설정
    time = np.linspace(0, duration, num_frames)  # 시작에서 끝까지 균일하게 시간을 분배

    # 꺾은선 그래프 그리기
    plt.figure(figsize=(12, 8))
    
    # 사용 가능한 요소만 그래프로 표시
    elements = ["sentence", "subject", "verb", "object", "prepositional phrase"]
    colors = ["gray", "green", "red", "purple", "orange"]  # 각 요소별 색상 설정
    for elem, color in zip(elements, colors):
        if elem in similarity_scores:  # 해당 요소가 유사도 데이터에 존재하는지 확인
            plt.plot(time, similarity_scores[elem], label=elem.capitalize(), color=color)

    # Timestamp 구간 표시
    start = timestamps[0]
    end = timestamps[1]
    plt.axvspan(start, end, color='grey', alpha=0.5, label='Timestamp')

    prop_start = proposal[0]
    prop_end = proposal[1]
    plt.axvspan(prop_start, prop_end, color='yellow', alpha=0.5, label='Proposal')

    # 그래프 제목과 축 레이블
    plt.title(f"Similarity over Time\nVideo ID: {video_id}, Sentence: {sentence}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Similarity")
    plt.legend()

    plt.subplots_adjust(bottom=0.25)

    timestamp_str = f"Timestamps: {timestamps}"
    proposal_str = f"Proposal: {proposal}"
    plt.figtext(0.9, 0.04, timestamp_str, wrap=True, horizontalalignment='right', fontsize=10, color='gray')
    plt.figtext(0.9, 0.01, proposal_str, wrap=True, horizontalalignment='right', fontsize=10, color='orange')

    # 하단 텍스트 라벨에 각 요소 정보 추가
    element_texts = [f"{elem.capitalize()}: {elements_obj.get(elem, 'None')}" for elem in ['subject', 'verb', 'object', 'prepositional phrase']]
    for idx, element_text in enumerate(element_texts, start=1):
        plt.figtext(0.5, 0.1 - idx*0.02, element_text, wrap=True, horizontalalignment='center', fontsize=10, color='black')
        
    # 그래프 저장
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 디렉토리가 없으면 생성
    save_path = os.path.join(save_dir, f"{video_id}_sentence_{sentence_index}.png")
    plt.savefig(save_path)  # 이미지 파일로 저장
    plt.close()

def plot_revised_and_replace_similarity_graph(video_id, sentence_index, similarities, save_dir):
    # 유사도 데이터를 가져옴
    sentence_data = similarities[video_id][sentence_index]
    
    # 문장과 관련된 메타데이터
    sentence = sentence_data["sentence"]
    duration = sentence_data["duration"]
    timestamps = sentence_data["timestamps"]
    proposal = sentence_data["proposal"][0]
    similarity_scores = sentence_data["similarity_scores"]
    subject_replaced = sentence_data['subject_replaced']
    verb_replaced = sentence_data['verb_replaced']
    object_replaced = sentence_data['object_replaced']
    
    # 시간축 설정: 프레임 수에 따라 전체 duration을 나눔
    num_frames = len(next(iter(similarity_scores.values())))  # 첫 번째 유효한 값의 길이로 프레임 수 설정
    time = np.linspace(0, duration, num_frames)  # 시작에서 끝까지 균일하게 시간을 분배

    # 꺾은선 그래프 그리기
    plt.figure(figsize=(12, 8))
    
    plt.plot(time, similarity_scores['revised_query'], label='revised_query'.capitalize(), color='black')
    
    # 사용 가능한 요소만 그래프로 표시
    linestyles = ['-', '--', '-.', ':']
    elements = ["subject_replaced", "verb_replaced", "object_replaced"]
    colors = ["green", "red", "purple"]  # 각 요소별 색상 설정
    for elem, color in zip(elements, colors):
        if elem in similarity_scores:
            for idx, similarity_score in enumerate(similarity_scores[elem]):
                label = f"{elem.capitalize()} {idx + 1}"  # 라벨에 텍스트 추가
                linestyle = linestyles[idx % len(linestyles)]  # 스타일 변경
                plt.plot(time, similarity_score, label=label, color=color, linestyle=linestyle, alpha=0.5)
        else:
            print(f"Warning: {elem} not found in similarity_scores.")

                
    # Timestamp 구간 표시
    start = timestamps[0]
    end = timestamps[1]
    plt.axvspan(start, end, color='grey', alpha=0.5, label='Timestamp')

    prop_start = proposal[0]
    prop_end = proposal[1]
    plt.axvspan(prop_start, prop_end, color='yellow', alpha=0.5, label='Proposal')

    # 그래프 제목과 축 레이블
    plt.title(f"Similarity over Time\nVideo ID: {video_id}, Sentence: {sentence}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Similarity")
    plt.legend()

    plt.subplots_adjust(bottom=0.25)

    timestamp_str = f"Timestamps: {timestamps}"
    proposal_str = f"Proposal: {proposal}"
    plt.figtext(0.9, 0.04, timestamp_str, wrap=True, horizontalalignment='right', fontsize=10, color='gray')
    plt.figtext(0.9, 0.01, proposal_str, wrap=True, horizontalalignment='right', fontsize=10, color='orange')
    
    step = 0
    for idx, text in enumerate(subject_replaced):
       plt.figtext(0.1, 0.16 - step, "subject_replaced" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016
    for idx, text in enumerate(verb_replaced):
       plt.figtext(0.1, 0.16 - step, "verb_replaced" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016
    for idx, text in enumerate(object_replaced):
       plt.figtext(0.1, 0.16 - step, "object_replaced" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016

    # 그래프 저장
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 디렉토리가 없으면 생성
    save_path = os.path.join(save_dir, f"{video_id}_sentence_{sentence_index}.png")
    plt.savefig(save_path)  # 이미지 파일로 저장
    plt.close()

def plot_masked_similarity_graph(video_id, sentence_index, similarities, save_dir):
    # 유사도 데이터를 가져옴
    sentence_data = similarities[video_id][sentence_index]
    
    # 문장과 관련된 메타데이터
    sentence = sentence_data["sentence"]
    duration = sentence_data["duration"]
    timestamps = sentence_data["timestamps"]
    proposal = sentence_data["proposal"][0]
    similarity_scores = sentence_data["similarity_scores"]
    subject_masked = sentence_data['subject_masked']
    verb_masked = sentence_data['verb_masked']
    object_masked = sentence_data['object_masked']
    prepositional_phrase_masked = sentence_data['prepositional_phrase_masked']
    
    # 시간축 설정: 프레임 수에 따라 전체 duration을 나눔
    num_frames = len(next(iter(similarity_scores.values())))  # 첫 번째 유효한 값의 길이로 프레임 수 설정
    time = np.linspace(0, duration, num_frames)  # 시작에서 끝까지 균일하게 시간을 분배

    # 꺾은선 그래프 그리기
    plt.figure(figsize=(12, 8))
    
    plt.plot(time, similarity_scores['revised_query'], label='revised_query'.capitalize(), color='black')
    
    # 사용 가능한 요소만 그래프로 표시
    linestyles = ['-', '--', '-.', ':']
    elements = ["subject_masked", "verb_masked", "object_masked", "prepositional_phrase_masked"]
    colors = ["green", "red", "purple", "orange"]  # 각 요소별 색상 설정
    for elem, color in zip(elements, colors):
        if elem in similarity_scores:
            for idx, similarity_score in enumerate(similarity_scores[elem]):
                label = f"{elem.capitalize()} {idx + 1}"  # 라벨에 텍스트 추가
                linestyle = linestyles[idx % len(linestyles)]  # 스타일 변경
                plt.plot(time, similarity_score, label=label, color=color, linestyle=linestyle, alpha=0.5)
        else:
            print(f"Warning: {elem} not found in similarity_scores.")

                
    # Timestamp 구간 표시
    start = timestamps[0]
    end = timestamps[1]
    plt.axvspan(start, end, color='grey', alpha=0.5, label='Timestamp')

    prop_start = proposal[0]
    prop_end = proposal[1]
    plt.axvspan(prop_start, prop_end, color='yellow', alpha=0.5, label='Proposal')

    # 그래프 제목과 축 레이블
    plt.title(f"Similarity over Time\nVideo ID: {video_id}, Sentence: {sentence}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Similarity")
    plt.legend()

    plt.subplots_adjust(bottom=0.25)

    timestamp_str = f"Timestamps: {timestamps}"
    proposal_str = f"Proposal: {proposal}"
    plt.figtext(0.9, 0.04, timestamp_str, wrap=True, horizontalalignment='right', fontsize=10, color='gray')
    plt.figtext(0.9, 0.01, proposal_str, wrap=True, horizontalalignment='right', fontsize=10, color='orange')
    
    step = 0
    for idx, text in enumerate(subject_masked):
       plt.figtext(0.1, 0.16 - step, "subject_masked" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016
    for idx, text in enumerate(verb_masked):
       plt.figtext(0.1, 0.16 - step, "verb_masked" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016
    for idx, text in enumerate(object_masked):
       plt.figtext(0.1, 0.16 - step, "object_masked" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016
    for idx, text in enumerate(prepositional_phrase_masked):
       plt.figtext(0.1, 0.16 - step, "prepositional_phrase_masked" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016

    # 그래프 저장
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 디렉토리가 없으면 생성
    save_path = os.path.join(save_dir, f"{video_id}_sentence_{sentence_index}.png")
    plt.savefig(save_path)  # 이미지 파일로 저장
    plt.close()

def plot_masked_important_similarity_graph(video_id, sentence_index, similarities, save_dir):
    # 유사도 데이터를 가져옴
    sentence_data = similarities[video_id][sentence_index]
    
    # 문장과 관련된 메타데이터
    sentence = sentence_data["sentence"]
    duration = sentence_data["duration"]
    timestamps = sentence_data["timestamps"]
    proposal = sentence_data["proposal"][0]
    similarity_scores = sentence_data["similarity_scores"]
    subject_masked = sentence_data['subject_masked']
    verb_masked = sentence_data['verb_masked']
    object_masked = sentence_data['object_masked']
    prepositional_phrase_masked = sentence_data['prepositional_phrase_masked']
    
    # 시간축 설정: 프레임 수에 따라 전체 duration을 나눔
    num_frames = len(next(iter(similarity_scores.values())))  # 첫 번째 유효한 값의 길이로 프레임 수 설정
    time = np.linspace(0, duration, num_frames)  # 시작에서 끝까지 균일하게 시간을 분배

    # 꺾은선 그래프 그리기
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 왼쪽 y축 (revised_query)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Revised Query Similarity", color='black')
    # revised_query 유사도 곡선 그리기
    revised_query_similarity = np.array(similarity_scores['revised_query'])  # NumPy 배열로 변환
    ax1.plot(time, revised_query_similarity, label='revised_query'.capitalize(), color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # 오른쪽 y축 설정
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Masked Elements Similarity', color='blue')  

    # 사용 가능한 요소만 그래프로 표시
    linestyles = ['-', '--', '-.', ':']
    elements = ["subject_masked", "verb_masked", "object_masked", "prepositional_phrase_masked"]
    colors = ["green", "red", "purple", "orange"]  # 각 요소별 색상 설정
    for elem, color in zip(elements, colors):
        if elem in similarity_scores:
            for idx, similarity_score in enumerate(similarity_scores[elem]):
                label = f"{elem.capitalize()} {idx + 1}"  # 라벨에 텍스트 추가
                linestyle = linestyles[idx % len(linestyles)]  # 스타일 변경
                similarity_score_array = np.array(similarity_score)
                adjusted_similarity = revised_query_similarity - similarity_score_array
                ax2.plot(time, adjusted_similarity, label=label, color=color, linestyle=linestyle, alpha=0.5)
        else:
            print(f"Warning: {elem} not found in similarity_scores.")
    
    ax2.tick_params(axis='y', labelcolor='blue')

    # Timestamp 구간 표시
    start = timestamps[0]
    end = timestamps[1]
    ax1.axvspan(start, end, color='grey', alpha=0.5, label='Timestamp')

    prop_start = proposal[0]
    prop_end = proposal[1]
    ax1.axvspan(prop_start, prop_end, color='yellow', alpha=0.5, label='Proposal')

    # 그래프 제목과 축 레이블
    plt.title(f"Similarity over Time\nVideo ID: {video_id}, Sentence: {sentence}")
    fig.tight_layout()  

    # 범례 표시
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Similarity")
    plt.legend()

    plt.subplots_adjust(bottom=0.25)

    timestamp_str = f"Timestamps: {timestamps}"
    proposal_str = f"Proposal: {proposal}"
    plt.figtext(0.9, 0.04, timestamp_str, wrap=True, horizontalalignment='right', fontsize=10, color='gray')
    plt.figtext(0.9, 0.01, proposal_str, wrap=True, horizontalalignment='right', fontsize=10, color='orange')
    
    step = 0
    for idx, text in enumerate(subject_masked):
       plt.figtext(0.1, 0.16 - step, "subject_masked" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016
    for idx, text in enumerate(verb_masked):
       plt.figtext(0.1, 0.16 - step, "verb_masked" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016
    for idx, text in enumerate(object_masked):
       plt.figtext(0.1, 0.16 - step, "object_masked" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016
    for idx, text in enumerate(prepositional_phrase_masked):
       plt.figtext(0.1, 0.16 - step, "prepositional_phrase_masked" + str(idx) + ": " + text, wrap=True, horizontalalignment='left', fontsize=9, color='gray')
       step += 0.016

    # 그래프 저장
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 디렉토리가 없으면 생성
    save_path = os.path.join(save_dir, f"{video_id}_sentence_{sentence_index}.png")
    plt.savefig(save_path)  # 이미지 파일로 저장
    plt.close()

# JSON 파일 로드
with open('./similarities_per_frame-charades-masked-align_indices.json', 'r', encoding='utf-8') as f:
    similarities = json.load(f)

# 저장 디렉토리 지정
save_directory = "./similarity_plots-charades-masked-align_indices"

random_samples = [(vid, i) for vid in similarities for i in range(len(similarities[vid]))]
# random_selected = random.sample(random_samples, 1)
for video_id, sentence_index in random_samples:
    plot_masked_similarity_graph(video_id, sentence_index, similarities, save_directory)


# 원하는 비디오 ID와 sentence 인덱스를 입력받아 플롯을 그릴 수 있도록
# selected = [('v_7QxUtHqQdbY', 1), ('v_twrPZghmNtA', 0), ('v_6g80a1NnftU', 2), ('v_30Yk_1Yc7Vk', 2), ('v_F30odTEdsxo', 2), ('v_6LLDsbc8XMM', 0), ('v_AO-0r8H2DOo', 0), ('v_cdpPn-7R3GQ', 0), ('v_hmb86jpgWfE', 0), ('v_HtkuvF7VbSQ', 2), ('v_KFS_lGlO-Ew', 0), ('v_yACg55C3IlM', 0), ]
# for video_id, sentence_index in selected:
#     plot_similarity_graph(video_id, sentence_index, similarities, save_directory)
