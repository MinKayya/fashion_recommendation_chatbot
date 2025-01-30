import os
import pandas as pd
import openai
import faiss
import numpy as np
import requests
import streamlit as st
import psutil
import time

# API 키 보호 (환경변수 사용 권장)
OPENWEATHER_API_KEY = "****"
OPENAI_API_KEY = "****"
openai.api_key = OPENAI_API_KEY

# OpenWeatherMap API URL 설정
CITY = "Seoul"
WEATHER_API_URL = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"

# 미리 로드할 CSV 및 FAISS 인덱스 파일 경로
DATA_PATH = "proff.csv"
INDEX_PATH = "fashion.index"

def get_weather_data():
    """현재 날씨 정보를 가져오는 함수"""
    response = requests.get(WEATHER_API_URL)
    weather_data = response.json()
    return {
        "temperature": weather_data["main"]["temp"],
        "wind_speed": weather_data["wind"]["speed"],
        "precipitation": weather_data.get("rain", {}).get("1h", 0)
    }

def create_or_load_faiss_index():
    """FAISS 인덱스를 로드하거나 새롭게 생성하는 함수"""
    if os.path.exists(INDEX_PATH):
        df = pd.read_csv(DATA_PATH)
        index = faiss.read_index(INDEX_PATH)
    else:
        df = pd.read_csv(DATA_PATH)
        required_columns = {"스타일", "카테고리", "색상", "기장", "소재", "핏"}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV 파일에 필요한 열이 없습니다. '스타일', '카테고리', '색상', '기장', '소재', '핏' 열이 포함되어야 합니다.")
        
        df['embedding_text'] = df.apply(lambda x: f"{x['스타일']} {x['카테고리']} {x['색상']} {x['기장']} {x['소재']} {x['핏']}", axis=1)
        embeddings = [
            openai.Embedding.create(input=text, model="text-embedding-ada-002")['data'][0]['embedding']
            for text in df['embedding_text']
        ]

        embeddings = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
    
    return df, index

def recommend_fashion(user_request, df, index):
    """사용자의 요청을 기반으로 적절한 패션을 추천하는 함수"""

    # 사용자 요청 임베딩 생성
    response = openai.Embedding.create(input=user_request, model="text-embedding-ada-002")
    user_embedding = np.array(response['data'][0]['embedding']).astype("float32").reshape(1, -1)
    D, I = index.search(user_embedding, k=10)
    recommendations = df.iloc[I[0]].to_dict(orient="records")

    weather_data = get_weather_data()
    
    # 추천 조합 생성
    tops, bottoms, dresses, outerwears = [], [], [], []
    for rec in recommendations:
        category = rec['카테고리']
        if category == "드레스":
            dresses.append(rec)
        elif category == "상의":
            tops.append(rec)
        elif category == "하의":
            bottoms.append(rec)
        elif category == "아우터":
            outerwears.append(rec)

    # 추천 메시지 생성
    prompt = f"현재 날씨는 기온 {weather_data['temperature']}°C, 바람 {weather_data['wind_speed']} m/s, 강수량 {weather_data['precipitation']} mm 입니다.\n"
    prompt += f"사용자가 '{user_request}' 라고 요청했습니다.\n\n추천 조합:\n"

    if dresses:
        for dress in dresses:
            prompt += f"- 드레스: {dress['스타일']}, 색상: {dress['색상']}, 소재: {dress['소재']}, 핏: {dress['핏']}\n"
            if outerwears:
                prompt += "  아우터 포함:\n"
                for outer in outerwears:
                    prompt += f"  - {outer['스타일']}, 색상: {outer['색상']}, 소재: {outer['소재']}, 핏: {outer['핏']}\n"
    elif tops and bottoms:
        for top in tops:
            prompt += f"- 상의: {top['스타일']}, 색상: {top['색상']}, 소재: {top['소재']}, 핏: {top['핏']}\n"
            for bottom in bottoms:
                prompt += f"  하의: {bottom['스타일']}, 색상: {bottom['색상']}, 소재: {bottom['소재']}, 핏: {bottom['핏']}\n"
            if outerwears:
                prompt += "  아우터 포함:\n"
                for outer in outerwears:
                    prompt += f"  - {outer['스타일']}, 색상: {outer['색상']}, 소재: {outer['소재']}, 핏: {outer['핏']}\n"

    # LLM을 사용하여 최종 추천 생성
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful fashion recommendation assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=700
    )
    
    return response['choices'][0]['message']['content'].strip()

# Streamlit 앱 시작
st.title("날씨 기반 RAG 패션 추천 시스템")

user_request = st.text_input("패션 추천 요청을 입력하세요 (예: '오늘 날씨에 맞는 캐주얼한 옷을 추천해줘')")

try:
    df, index = create_or_load_faiss_index()
except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

if st.button("추천 받기"):
    if user_request:
        recommendation = recommend_fashion(user_request, df, index)
        st.subheader("추천 결과")
        st.write(recommendation)
    else:
        st.warning("추천 요청을 입력하세요.")
