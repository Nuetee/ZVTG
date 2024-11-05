import json

with open('./dataset/activitynet/llm_outputs.json') as f:
    data = json.load(f)

filtered_data = {}
for key, value in data.items():
    responses = value.get("response", [])
    filtered_responses = []
    filtered_timestamps = []
    filtered_sentences = []
    
    for idx, response in enumerate(responses):
        # if response.get("relationship") != "simultaneously" and response.get("relationship") != "sequentially":
        if response.get("relationship") != "single-query":
            filtered_responses.append(response)
            filtered_timestamps.append(value["timestamps"][idx])
            filtered_sentences.append(value["sentences"][idx])
    
    if filtered_responses:
        filtered_data[key] = {
            "duration": value["duration"],
            "timestamps": filtered_timestamps,
            "sentences": filtered_sentences,
            "response": filtered_responses
        }

# 결과를 새로운 JSON 파일로 저장
with open('./dataset/activitynet/llm_outputs_single-query.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)