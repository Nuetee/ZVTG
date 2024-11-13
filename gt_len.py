import json
from collections import Counter

# JSON 파일을 로드하는 함수 (파일 경로를 지정해 주세요)
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# timestamps의 길이를 구하고 사용자 정의 구간별로 분포를 계산하는 함수
def get_custom_distribution(data, bins):
    lengths = []

    # 모든 timestamps 길이 계산
    for video_id, video_data in data.items():
        timestamps = video_data.get("timestamps", [])
        for start, end in timestamps:
            length = end - start
            lengths.append(length)
    
    # 각 구간별로 개수 집계
    distribution = {f"{bin_}초 미만": 0 for bin_ in bins}
    for length in lengths:
        for bin_ in bins:
            if length < bin_:
                distribution[f"{bin_}초 미만"] += 1
                break  # 첫 번째 해당 구간에만 포함시키고 종료

    # 누적 개수로 수정 (이전 구간들의 개수를 합산하여 첫 코드와 일치시키기 위함)
    cumulative_distribution = {}
    cumulative_count = 0
    for bin_ in bins:
        cumulative_count += distribution[f"{bin_}초 미만"]
        cumulative_distribution[f"{bin_}초 미만"] = cumulative_count

    return cumulative_distribution

# 파일 경로와 사용자 정의 구간 설정
file_path = './dataset/charades-sta/llm_outputs.json'
# file_path = './dataset/activitynet/llm_outputs.json'
bins = [5, 15, 20, 25]  # 원하는 구간을 리스트로 정의

# 실행
data = load_json(file_path)
distribution = get_custom_distribution(data, bins)

# 결과 출력
for interval, count in distribution.items():
    print(f"{interval}: {count}개")
