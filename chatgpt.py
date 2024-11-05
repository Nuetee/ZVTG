from openai import OpenAI
import os
import json
from tqdm import tqdm

def parsing_query(client, prompt_text, llm_outputs_path, save_path):
    with open(llm_outputs_path) as f:
        data = json.load(f)

    for key, value in tqdm(data.items()):
        sentences = value.get("sentences", [])
        response = value.get("response")

        for idx, sentence in enumerate(sentences):
            user_content = prompt_text + "\ninput query: \"" + sentence + "\"\n Parsed result: "
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_content }
                ]
            )
            
            if "parsed_query" not in value:
                value["parsed_query"] = []
            
            parsed_content = completion.choices[0].message.content
            cleaned_content = parsed_content.replace("```python", "").replace("```", "").strip()
            
            try:
                parsed_query = json.loads(parsed_content)
            except json.JSONDecodeError:
                print(f"Error parsing content: {parsed_content}")
                continue  # 스킵하고 다음 문장으로 넘어가기
            # parsed_query 배열에 API 결과를 추가 (append)
            value["parsed_query"].append(parsed_query)

            if 'query_json' in response[idx]:
                for j in range(0, len(response[idx]['query_json'])):
                    for query in response[idx]['query_json'][j]['descriptions']:
                        user_content = prompt_text + "\ninput query: \"" + query + "\"\n Parsed result: "
                        completion = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": user_content }
                            ]
                        )
                        if "parsed_query" not in value['response'][idx]['query_json'][j]:
                            value['response'][idx]['query_json'][j]["parsed_query"] = []
                        
                        parsed_content = completion.choices[0].message.content
                        cleaned_content = parsed_content.replace("```python", "").replace("```", "").strip()
                
                        try:
                            parsed_query = json.loads(parsed_content)
                        except json.JSONDecodeError:
                            print(f"Error parsing content: {parsed_content}")
                            continue  # 스킵하고 다음 문장으로 넘어가기
                        # parsed_query 배열에 API 결과를 추가 (append)
                        value['response'][idx]['query_json'][j]["parsed_query"].append(parsed_query)

    # 변경된 데이터를 새로운 파일로 저장
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def revise_and_replace(client, prompt, source_path, save_path):
    with open(source_path) as f:
        data = json.load(f)

    for key, value in tqdm(data.items()):
        sentences = value.get("sentences", [])

        for idx, sentence in enumerate(sentences):
            user_content = "User input: \"" + sentence + "\""
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content }
                ]
            )
            
            if "revise_and_replace" not in value:
                value["revise_and_replace"] = []

            content = completion.choices[0].message.content
            
            try:
                revise_and_replace = json.loads(content)
            except json.JSONDecodeError:
                print(f"Error parsing content: {content}")
                continue
            
            value["revise_and_replace"].append(revise_and_replace)


def masking(client, prompt, source_path, save_path):
    with open(source_path) as f:
        data = json.load(f)

    for key, value in tqdm(data.items()):
        revise_and_replace = value.get("revise_and_replace", [])
         # 'response' 키 삭제
        if 'response' in value:
            del value['response']

        for idx, sentence in enumerate(revise_and_replace):
            revised_query = sentence['revised_query'][0]
            user_content = "User input: \"" + revised_query + "\""
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content }
                ]
            )
            
            if "masked" not in value:
                value["masked"] = []

            content = completion.choices[0].message.content
            
            try:
                masked = json.loads(content)
            except json.JSONDecodeError:
                print(f"Error parsing content: {content}")
                continue
            value["masked"].append(masked)
    # 변경된 데이터를 새로운 파일로 저장
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__=='__main__':    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "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"))
    # region
    # prompt_text = """
    # Task:
    # 1. Analyze the given input query by identifying the key components: subject, verb, object, and prepositional phrase.
    # 2. When parsing:
    #     - Include any adverbs as part of the verb.
    #     - Group prepositional phrases that modify the subject with the subject.
    #     - If the prepositional phrase modifies the verb or another part of the sentence, list it separately as a prepositional phrase.
    #     - If there is no object or prepositional phrase, set them as "None".
    # 3. Finally, return the parsed query as a **valid JSON object** with the keys: subject, verb, object, and prepositional phrase. Do not include any code block or Python syntax.

    # Examples:
    # 1.  - Input query: "A woman in black coat walks outside."
    #     - Parsed result: {"subject": "A woman in black coat", "verb": "walks outside", "object": "None", "prepositional phrase": "None"}
    # 2.  - Input query: "A young lady is gripping a black and silver punching bag between her legs."
    #     - Parsed result: {"subject": "A young lady", "verb": "is gripping", "object": "a black and silver punching bag", "prepositional phrase": "between her legs"}
    # 3. - Input query: "The man runs quickly through the park."
    #     - Parsed result: {"subject": "The man", "verb": "runs quickly", "object": "None", "prepositional phrase": "through the park"}
    # 4.  - Input query: "A man holds food and drink in the car."
    #     - Parsed result: {"subject": "A man", "verb": "holds", "object": "food and drink", "prepositional phrase": "in the car"}


    # Instructions:
    # Please parse the following input query and generate a Python dictionary with the keys: subject, verb, object, and prepositional phrase.
    # """
    # # parsing
    # # parsing_query(client, './dataset/activitynet/llm_outputs.json', './dataset/activitynet/llm_outputs-parsed_query.json')
    
    # prompt_text = """
    # Your task involves three specific actions when processing user text input describing human activities in a video:

    # Fix unnatural phrasing: If the user input sounds awkward or unnatural, revise it to make the phrasing more natural.
    # Simplify complex expressions: If the user input contains complex expressions, simplify them into clearer and more straightforward language.
    # Replace difficult or uncommon words: If the user input uses difficult or uncommon words, replace them with more commonly used and simpler words.
    # If the user input is already natural, free of complex expressions, and uses commonly understood words, you do not need to make any changes. In that case, return the user input as is.

    # Examples:

    # - User input: "Several people slide down, a woman and her little girl."
    # - Revised input: "Several people, including a woman and her little girl, slide down."

    # - User input: "One of the girl then walks away and goes to get something from a man cleaning the motorcycle."
    # - Revised input: "One of the girls then walks away to get something from a man cleaning the motorcycle."

    # - User input: "She sprays it with a spray bottle and continues brushing her hair."
    # - Revised input: "She uses a spray bottle and brushes her hair."
    # """
    # revise_query(client, prompt_text, './dataset/activitynet/llm_outputs.json', './dataset/activitynet/llm_outputs-revised_query.json')

    
    
    # prompt_text = """
    # A user text input describing human activities in a video will be provided. Your task is to create counterfactual sentences by altering the meaning of the subject, verb, and object in the user text input. For each part of speech (subject, verb, object), generate two counterfactuals.

    # You should only respond in JSON format as described below:

    # **INSTRUCTIONS OF OUTPUTS:**

    # Your outputs should contain `"query_json": "<query_json>"`.  
    # Example:  
    # User input: `"person sits on the table"`

    # ```json
    # "counterfactual_query": {
    #     "subject": ["dog sits on the table", "a book sits on the table"],
    #     "verb": ["person stands on the table", "person jumps on the table"],
    #     "object": ["person sits on the chair", "person sits on the bench"]
    # }
    # ```

    # User input: `"person eating a sandwich"`

    # ```json
    # "counterfactual_query": {
    #     "subject": ["dog is eating a sandwich", "child is eating a sandwich"],
    #     "verb": ["person is holding a sandwich", "person is making a sandwich"],
    #     "object": ["person is eating a salad", "person is eating a burger"]
    # }
    # ```

    # User input: `"person drinks water from a cup"`

    # ```json
    # "counterfactual_query": {
    #     "subject": ["cat drinks water from a cup", "dog drinks water from a cup"],
    #     "verb": ["person fills the cup with water", "person pours water from a cup"],
    #     "object": ["person drinks juice from a cup", "person drinks water from a bottle"]
    # }
    # ```

    # User input: `"person opened up the refrigerator"`

    # ```json
    # "counterfactual_query": {
    #     "subject": ["dog opened up the refrigerator", "the robot opened up the refrigerator"],
    #     "verb": ["person closed the refrigerator", "person pushed the refrigerator"],
    #     "object": ["person opened up the cabinet", "person opened up the freezer"]
    # }
    # ```
    # """
    # counterfactual_query(client, prompt_text, './dataset/activitynet/llm_outputs.json', './dataset/activitynet/llm_outputs-revised_query.json')
    # endregion


    prompt_masking_task = """
    Your task involves recognizing the subject, verb, object, and prepositional phrases in user text input describing human activities in a video. You will mask these elements with [MASK] one at a time and return the corresponding sentences. If the sentence contains more than one subject, verb, object, or prepositional phrase, return a separate sentence with each element masked in turn.

    1. **Identify and mask the following elements:**
    - **Subject masking:** Mask the subject (and any adjectives or adjective phrases that describe the subject) with [MASK].
    - **Verb masking:** Mask the verb with [MASK].
    - **Object masking:** Mask the object (and any adjectives or adjective phrases that describe the object) with [MASK].
    - **Prepositional phrase masking:** Mask the entire prepositional phrase (e.g., "near the tree") with [MASK].

    2. **Handling multiple instances:**
    - If there is more than one subject, verb, object, or prepositional phrase in a sentence, return multiple sentences, each with only one element masked.

    3. **Output format:** 
    - Ensure the output is valid JSON.
    - Do not include any extra formatting like code blocks or tags.
    - Do not include any extra characters like ```json or similar markers.

    Output Format:
    {
        "subject_masked": ["Sentence with one subject masked."],
        "verb_masked": ["Sentence with one verb masked."],
        "object_masked": ["Sentence with one object masked."],
        "prepositional_phrase_masked": ["Sentence with one prepositional phrase masked."]
    }

    **Example 1:**
    User input: "The tall man is throwing a ball to the child near the tree."

    Output:
    {
        "subject_masked": [ "[MASK] is throwing a ball to the child near the tree." ],
        "verb_masked": [ "The tall man is [MASK] a ball to the child near the tree." ],
        "object_masked": [ "The tall man is throwing [MASK] to the child near the tree." ],
        "prepositional_phrase_masked": [ "The tall man is throwing a ball to the child [MASK]." ]
    }

    **Example 2:**
    User input: "John and Mary are riding their bikes through the park."

    Output:
    {
        "subject_masked": [ "[MASK] and Mary are riding their bikes through the park.", "John and [MASK] are riding their bikes through the park." ],
        "verb_masked": [ "John and Mary are [MASK] their bikes through the park." ],
        "object_masked": [ "John and Mary are riding [MASK] through the park." ],
        "prepositional_phrase_masked": [ "John and Mary are riding their bikes [MASK]." ]
    }

    **Example 3:**
    User input: "A cat is sitting on the table under the lamp."

    Output:
    {
        "subject_masked": [ "[MASK] is sitting on the table under the lamp." ],
        "verb_masked": [ "A cat is [MASK] on the table under the lamp." ],
        "object_masked": [ "A cat is sitting on [MASK] under the lamp." ],
        "prepositional_phrase_masked": [ "A cat is sitting [MASK] under the lamp.", "A cat is sitting on the table [MASK]." ]
    }

    **Example 4:**
    User input: "The man picked up the phone and called his friend."

    Output:
    {
        "subject_masked": [ "[MASK] picked up the phone and called his friend." ],
        "verb_masked": [ "The man [MASK] the phone and called his friend.", "The man picked up the phone and [MASK] his friend." ],
        "object_masked": [ "The man picked up [MASK] and called his friend.", "The man picked up the phone and called [MASK]." ],
        "prepositional_phrase_masked": []
    }

    **Example 5:**
    User input: "She baked a cake and decorated it with icing."

    Output:
    {
        "subject_masked": [ "[MASK] baked a cake and decorated it with icing." ],
        "verb_masked": [ "She [MASK] a cake and decorated it with icing.", "She baked a cake and [MASK] it with icing." ],
        "object_masked": [ "She baked [MASK] and decorated it with icing.", "She baked a cake and decorated [MASK] with icing." ],
        "prepositional_phrase_masked": [ "She baked a cake and decorated it [MASK]." ]
    }
    """
    masking(client, prompt_masking_task, './dataset/activitynet/llm_outputs-revise_and_replace.json', './dataset/activitynet/llm_outputs-masked.json')