import json
import matplotlib.pyplot as plt

def extract_durations_from_jsonl(jsonl_file):
    """
    JSONL 파일에서 'duration' 값을 추출하는 함수
    """
    durations = []
    gt_ratio = []

    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:  # JSONL 파일은 한 줄씩 JSON 객체로 읽음
            data = json.loads(line.strip())  # JSON 변환
            # if "duration" in data and isinstance(data["duration"], (int, float)):  
            #     durations.append(data["duration"])
            # gt.append(data["relevant_windows"])
            duration = data["duration"]
            gts = data["relevant_windows"]
            for gt in gts:
                len = gt[1] - gt[0]
                gt_ratio.append(len/duration)
            

    return gt_ratio

def save_histogram(durations, bins=10, filename="histogram.png"):
    """
    'duration' 값의 히스토그램을 생성하고 저장하는 함수 (X축 눈금 추가)
    """
    if not durations:
        print("No 'duration' values found.")
        return

    plt.figure(figsize=(10, 5))
    
    # 히스토그램 생성
    counts, bins, _ = plt.hist(durations, bins=bins, alpha=0.75, edgecolor="black")

    # X축 눈금을 bin 경계값으로 설정
    plt.xticks(bins, rotation=45)  

    # 라벨 및 제목 설정
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.title("Histogram of 'duration' Values")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # 히스토그램 저장 (300 DPI)
    plt.savefig(filename, dpi=300)
    plt.close()  # 메모리 해제

    print(f"Histogram saved as {filename}")

def main(jsonl_file):
    """
    JSONL 파일을 로드하고 'duration' 값의 히스토그램을 저장하는 메인 함수
    """
    durations = extract_durations_from_jsonl(jsonl_file)

    save_histogram(durations, bins=20)

# 사용 예제
if __name__ == "__main__":
    json_file_path = "dataset/qvhighlight/highlight_val_release.jsonl"  # JSON 파일 경로 입력
    main(json_file_path)
