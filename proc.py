import os
import json
import pandas as pd
from multiprocessing import Pool, cpu_count

# 각 JSON 파일을 처리하는 함수
def process_single_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            # 이미지 식별자 추출
            image_id = data["이미지 정보"]["이미지 식별자"]

            # 스타일 정보가 포함된 라벨링 정보 추출
            styles = data["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]["스타일"]

            # 카테고리별 데이터 추출
            categories = ["상의", "하의", "아우터", "원피스"]
            all_data = []
            for category in categories:
                if category in data["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"]:
                    items = data["데이터셋 정보"]["데이터셋 상세설명"]["라벨링"][category]
                    for item in items:
                        if item:  # 빈 객체가 아닌 경우에만 처리
                            item_data = {
                                "이미지 식별자": image_id,
                                "카테고리": category,
                                "스타일": styles[0].get("스타일", ""),
                                "서브스타일": styles[0].get("서브스타일", ""),
                                "색상": item.get("색상", ""),
                                "기장": item.get("기장", ""),
                                "옷깃": item.get("옷깃", ""),
                                "디테일": ', '.join(item.get("디테일", [])),
                                "소매기장": item.get("소매기장", ""),
                                "소재": ', '.join(item.get("소재", [])),
                                "프린트": ', '.join(item.get("프린트", [])),
                                "핏": item.get("핏", ""),
                            }
                            all_data.append(item_data)
            return all_data
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# 전체 폴더 내 모든 JSON 파일을 병렬 처리하는 함수
def process_all_json_files(main_folder, output_file="processed_fashion_data.csv", batch_size=100):
    all_files = []

    for root, dirs, files in os.walk(main_folder):
        for filename in files:
            if filename.endswith(".json"):
                all_files.append(os.path.join(root, filename))

    num_batches = len(all_files) // batch_size + (len(all_files) % batch_size > 0)
    with Pool(cpu_count()) as pool:
        for i in range(num_batches):
            batch_files = all_files[i * batch_size:(i + 1) * batch_size]
            batch_data = pool.map(process_single_json, batch_files)

            # 배치 데이터 저장
            batch_data = [item for sublist in batch_data for item in sublist]  # Flatten the list
            df = pd.DataFrame(batch_data)
            if i == 0:
                df.to_csv(output_file, mode="w", index=False, encoding="utf-8-sig")
            else:
                df.to_csv(output_file, mode="a", index=False, header=False, encoding="utf-8-sig")
            
            print(f"배치 {i + 1}/{num_batches} 처리 및 저장 완료")

    print(f"전처리된 파일이 '{output_file}'로 저장되었습니다.")

# 실행 예시
main_folder = "/mnt/c/ssd/llm/ffff"    # 최상위 폴더 경로 설정
output_file = "/mnt/c/ssd/llm/processed_fashion_data.csv"   # 출력 파일명
process_all_json_files(main_folder, output_file)
