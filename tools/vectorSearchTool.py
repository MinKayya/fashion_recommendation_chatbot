from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from config.settings import Settings


df = pd.read_csv(Settings.CSV_PATH)
index = faiss.read_index(Settings.INDEX_CONTEXT_PATH)
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def search(user_request, k=5):
    """
    사용자 요청 기반 추천 검색.
    :param user_request: 사용자가 입력한 요청 문자열.
    :param k: 추천할 항목 수.
    :return: 추천 세트 리스트.
    """
    
    # 사용자 요청을 임베딩 벡터로 변환
    embedding = embedding_model.encode(user_request)
    user_embedding = np.array(embedding).reshape(1, -1).astype('float32')

    # FAISS 인덱스에서 검색
    D, I = index.search(user_embedding, k)

    # 추천 세트 생성
    recommendations = []
    for idx in I[0]:
        if idx < 0:
            continue
        item = df.iloc[idx]
        image_id = item['이미지 식별자']
        set_items = df[df['이미지 식별자'] == image_id].to_dict(orient="records")
        recommendations.append({
            "이미지 식별자": image_id,
            "세트 아이템": set_items
        })

    # 중복 제거
    unique_recommendations = {rec["이미지 식별자"]: rec for rec in recommendations}
    return list(unique_recommendations.values())