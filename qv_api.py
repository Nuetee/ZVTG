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
    prompt = prompt = f"""
    You are given a user-provided text query describing a person's actions in a video. 
    Your task is to generate three variations of the query that meet the following requirements:

    ### Requirements:
    1. **Strictly Preserve Meaning**: The variations must maintain the exact same meaning as the original query. 
    Do not introduce any new or uncertain details that are not explicitly stated in the original query.
    2. **Diversify Sentence Structure and Word Choice**: The variations should rephrase the original query with 
    different syntax, vocabulary, and phrasing while keeping the meaning intact.
    3. **No Additional Information**: Do not add speculative or unrelated details. The reworded queries must 
    strictly adhere to the factual content of the original query.
    4. **Consistent with the Video Context**: Ensure that the variations do not introduce actions or objects 
    not mentioned in the original query.

    ### Output Format:
    Your response must be in valid JSON format, directly parsable by `json.loads()` in Python. 
    Do not include any additional explanations, markdown syntax, or extraneous characters. 
    The output format should be as follows:

    {{
        "variations": [
            "augmented_query1",
            "augmented_query2",
            "augmented_query3"
        ]
    }}

    ### Example Inputs and Outputs:

    #### Example 1
    **Input Query:**  
    "A man in a red shirt is picking up a box from the floor."

    **Expected Output:**
    {{
        "variations": [
            "A man wearing a red shirt lifts a box from the ground.",
            "A man in a red top is picking up a box from the floor.",
            "A person dressed in a red shirt bends down to grab a box."
        ]
    }}

    #### Example 2
    **Input Query:**  
    "A woman is running across a grassy field while holding a dog leash."

    **Expected Output:**
    {{
        "variations": [
            "A woman runs across a grassy field with a dog leash in her hand.",
            "A lady is sprinting over a grass-covered field while carrying a dog leash.",
            "A woman holding a dog leash is running through a grassy area."
        ]
    }}

    Generate three variations of the following input query while adhering to the rules mentioned above.

    **User Input Query:**  
    {query}
    """


    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        # 응답이 정상인지 확인
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            output_text = response.choices[0].message.content.strip()
            output_text = clean_json_output(output_text)  # 불필요한 텍스트 제거
            
            parsed_json = json.loads(output_text)
            if "variations" in parsed_json:
                return parsed_json
            print(f"⚠️ 'variations' 키가 누락된 응답: {parsed_json}")
        else:
            print("⚠️ OpenAI 응답이 비어 있습니다.")
            return {"variations": []}
    
    except Exception as e:
        print(f"⚠️ OpenAI API 호출 중 오류 발생: {e}")
        return {"variations": []}  # 오류 발생 시 빈 리스트 반환

def process_json_file(input_file, output_file):
    """JSON 파일을 읽고 query 변형을 추가한 후 저장"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    for entry in tqdm(data, desc="Processing queries", unit="query"):
        augmented_queries = generate_augmented_queries(entry["query"], num_variations=3)
        entry["query_augmented"] = augmented_queries["variations"]
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# 실행 예시
input_file = "dataset/qvhighlight/highlight_val_release.jsonl"  # 입력 JSON 파일 경로
output_file = "dataset/qvhighlight/highlight_val_release_aug.jsonl"  # 출력 JSON 파일 경로
process_json_file(input_file, output_file)
