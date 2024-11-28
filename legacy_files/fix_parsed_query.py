import json
import re

# 파이썬 딕셔너리 형태를 추출하는 정규식
pattern = r"query\s*=\s*({.*})"

def fix_parsed_queries(data):
    error_videos = []  # 에러가 발생한 비디오를 기록할 리스트
    error_count = 0  # 에러 발생한 비디오 개수 카운트
    
    # 원본 데이터를 변경하지 않기 위해 복사
    fixed_data = data.copy()

    for video_id, video_data in list(fixed_data.items()):  # items() 대신 list()로 사용하여 수정 중 제거 가능
        parsed_query_list = video_data.get('parsed_query', [])
        
        # 각 parsed_query 문자열에서 파이썬 딕셔너리 형태로 변환
        try:
            for idx, query_str in enumerate(parsed_query_list):
                # 정규식을 사용하여 딕셔너리 부분만 추출
                match = re.search(pattern, query_str, re.DOTALL)
                if match:
                    # 추출된 딕셔너리 문자열을 eval로 변환
                    parsed_query = eval(match.group(1))
                    video_data['parsed_query'][idx] = parsed_query
                else:
                    raise ValueError(f"No valid query found for video {video_id}")
        except Exception as e:
            print(f"Error parsing query for video {video_id}: {e}")
            error_videos.append(video_id)  # 에러가 발생한 비디오 ID 기록
            error_count += 1  # 에러 개수 카운트
            del fixed_data[video_id]  # 에러가 발생한 비디오를 데이터에서 제거

    return fixed_data, error_videos, error_count

# JSON 파일 로드
with open('./dataset/activitynet/llm_outputs_single-query-parsed.json') as f:
    data = json.load(f)

# parsed_query 값을 수정
fixed_data, error_videos, error_count = fix_parsed_queries(data)

# 변경된 데이터를 새로운 파일로 저장
with open('./dataset/activitynet/llm_outputs_single-query-parsed-fixed.json', 'w') as f:
    json.dump(fixed_data, f, indent=4)

# 에러 비디오 개수 출력
print(f"Error occurred in {error_count} videos.")
if error_videos:
    print(f"Videos with errors: {error_videos}")
