import json
import openai
from tqdm import tqdm
import re  # 정규 표현식을 사용하여 Markdown 코드 블록 제거

# OpenAI API 키 설정
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def clean_json_output(response_text):
    """응답에서 'User Input:' 및 'Expected Output:' 부분을 제거하고 JSON만 추출"""
    match = re.search(r"\{\s*\"variations\":.*\}", response_text, re.DOTALL)
    if match:
        return match.group(0)  # JSON만 추출하여 반환
    return response_text  # JSON이 없으면 원본 반환

def generate_augmented_queries(query, num_variations=3):
    """ChatGPT를 사용하여 query의 변형된 문장을 생성"""
    prompt = (
        "The user will input a sentence describing an activity in a video. Your task is to generate "
        f"{num_variations} different variations of the given sentence while preserving its original meaning.\n\n"
        "Instructions:\n"
        "- Maintain the original semantic meaning strictly.\n"
        "- Do not add any additional details that are not present in the original sentence.\n"
        "- Ensure that each variation has a different sentence structure and word usage.\n"
        "- The output should be formatted as a JSON array with three elements.\n\n"
        "Only return the JSON output. Do not include any additional text, such as 'User Input' or 'Expected Output'.\n\n"

        "Example:\n\n"
        "Example:\n"
        "{\n"
        "    \"variations\": [\n"
        "        \"A mother and her daughter prepared a meal while having a conversation over FaceTime.\",\n"
        "        \"While chatting on FaceTime, a girl and her mom were cooking together.\",\n"
        "        \"A girl and her mother engaged in cooking while video calling each other.\"\n"
        "    ]\n"
        "}"
        "Your response must be in JSON format and should be parsable using Python’s json.loads().\n"
        "Do not include any additional text or explanations. Only return the JSON output."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI that generates paraphrased sentences while preserving meaning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        # 응답이 정상인지 확인
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            output_text = response.choices[0].message.content.strip()
            output_text = clean_json_output(output_text)  # 불필요한 텍스트 제거
            try:
                parsed_json = json.loads(output_text)
                if "variations" in parsed_json:
                    return parsed_json  # 정상 응답 반환
                else:
                    print(f"⚠️ 'variations' 키가 누락된 응답: {parsed_json}")
                    return {"variations": []}  # variations가 없으면 빈 리스트 반환
            except json.JSONDecodeError:
                print(f"⚠️ JSON 디코딩 실패: {output_text}")
                return {"variations": []}  # 오류 발생 시 빈 리스트 반환
        else:
            print("⚠️ OpenAI 응답이 비어 있습니다.")
            return {"variations": []}
    
    except Exception as e:
        print(f"⚠️ OpenAI API 호출 중 오류 발생: {e}")
        return {"variations": []}  # 오류 발생 시 빈 리스트 반환

def process_json_file(input_file, output_file):
    import pdb;pdb.set_trace()
    """JSON 파일을 읽고 query 변형을 추가한 후 저장"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    pdb.set_trace()
    for entry in tqdm(data, desc="Processing queries", unit="query"):
        augmented_queries = generate_augmented_queries(entry["query"], num_variations=3)
        entry["query_augmented"] = augmented_queries["variations"]
    pdb.set_trace()
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# 실행 예시
input_file = "dataset/qvhighlight/highlight_val_release.jsonl"  # 입력 JSON 파일 경로
output_file = "dataset/qvhighlight/highlight_val_release_aug.jsonl"  # 출력 JSON 파일 경로
process_json_file(input_file, output_file)
