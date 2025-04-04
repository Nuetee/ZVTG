import json
import matplotlib.pyplot as plt

def extract_durations(data):
    """
    JSON 데이터에서 'duration' 키의 값을 재귀적으로 추출하는 함수
    """
    durations = []
    
    if isinstance(data, dict):  # 딕셔너리인 경우
        for key, value in data.items():
            if isinstance(value, dict):  # value가 다시 딕셔너리라면 탐색
                durations.append(value["duration"])
                
    return durations

def plot_histogram(durations, bins=10, filename="histogram.png"):
    """
    'duration' 값의 히스토그램을 생성하는 함수
    """
    if not durations:
        print("No 'duration' values found.")
        return

    plt.figure(figsize=(10, 5))
    counts, bins, _ = plt.hist(durations, bins=bins, alpha=0.75, edgecolor="black")

    # X축 눈금을 bin 경계값으로 설정
    plt.xticks(bins, rotation=45)  # X축 눈금을 bin 값으로 설정, 45도 기울이기

    # 각 범주에 데이터 개수 표시
    for count, bin_edge in zip(counts, bins[:-1]):  # 마지막 bin 제외
        plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, str(int(count)), 
                 ha="center", va="bottom", fontsize=10, color="black")
        
    # 라벨 및 제목 설정
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.title("Histogram of 'duration' Values")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # 히스토그램 저장 (300 DPI)
    plt.savefig(filename, dpi=300)
    plt.close()  # 메모리 해제

    print(f"Histogram saved as {filename}")

def main(json_file):
    """
    JSON 파일을 로드하고 'duration' 값의 히스토그램을 그리는 메인 함수
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    durations = extract_durations(data)
    # print(f"Extracted durations: {durations}")  # 디버깅용 출력
    plot_histogram(durations, bins=20)

# 사용 예제
if __name__ == "__main__":
    json_file_path = "dataset/activitynet/llm_outputs.json"  # JSON 파일 경로 입력
    main(json_file_path)
