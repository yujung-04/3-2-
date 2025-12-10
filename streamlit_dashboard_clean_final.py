import streamlit as st
import pandas as pd
import altair as alt 
import numpy as np
import os
import json 
import re
import joblib 

# 딥러닝 모델 통합을 위한 라이브러리
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# 🌟🌟🌟 Streamlit 설정은 모든 st. 명령 중 가장 먼저 실행되어야 합니다. 🌟🌟🌟
st.set_page_config(layout="wide")


# ====================================================================
# 0. 전역 설정 및 모델 경로 정의
# ====================================================================

# 🚨 1. 데이터 로드 경로 (통계용)
DATA_PATH = '/Users/sunkyong/Downloads/petitions/data_categorized/train_categorized.csv' 

# 🚨 2. 요약 모델 경로 (C 파트)
MODEL_LOCAL_PATH = "/Users/sunkyong/Downloads/petitions/final_models/kobart_summary_textrank_ft" 
device_sum = "cuda" if torch.cuda.is_available() else "cpu"

# 🚨 3. 분류 모델 에셋 디렉토리 (B 파트)
CLASSIFY_MODEL_DIR = "/Users/sunkyong/Downloads/petitions/classify" 
EMBEDDER_MODEL_NAME = "jhgan/ko-sroberta-multitask"
device_clf = "cuda" if torch.cuda.is_available() else "cpu"

# 최종 카테고리 매핑 딕셔너리
FULL_CATEGORY_MAP = {
    1: '과학기술/정보통신', 2: '교육', 3: '국토/해양/교통', 4: '기타',
    5: '농업/임업/수산업/축산업', 6: '문화/체육/관광/언론', 7: '보건의료', 8: '복지/보훈',
    9: '산업/통상', 10: '소비자/공정거래', 11: '수사/법무/사법제도', 12: '외교/통일/국방/안보',
    13: '인권/성평등/노동', 14: '재난/안전/환경', 15: '재정/세제/금융/예산', 16: '저출산/고령화/아동/청소년/가족',
    17: '정치/선거/국회운영', 18: '행정/지방자치'
}

# ====================================================================
# A. 모델 로드 함수 정의 (캐싱)
# ====================================================================

@st.cache_data
def load_data(file_path):
    # 통계 분석 데이터 로드
    df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
    if 'agree_count' in df.columns:
        df['agree_count'] = df['agree_count'].astype(str).str.replace(',', '', regex=False)
        df['agree_count'] = pd.to_numeric(df['agree_count'], errors='coerce').fillna(0).astype(int)
    category_map = {k: v for k, v in FULL_CATEGORY_MAP.items()}
    df['category_name'] = df['category'].map(category_map)
    return df

@st.cache_resource
def load_classification_assets(model_dir):
    # B 파트 분류 모델 에셋 로드
    
    # 🚨🚨🚨 임시 재저장 및 로드 로직 시작 🚨🚨🚨
    
    # 1. 원본 파일 로드 시도
    try:
        model = joblib.load(os.path.join(model_dir, 'classify_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        # 2. 로드 성공 시 현재 환경에 맞게 새 파일로 재저장
        st.warning("모델 로드 성공! 현재 환경 버전으로 '..._FIXED.pkl' 파일 재저장 중...")
        joblib.dump(model, os.path.join(model_dir, 'classify_model_FIXED.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler_FIXED.pkl'))
        st.success("✅ 모델 파일 재저장 완료! 이제 '..._FIXED.pkl'을 사용하도록 코드를 변경하세요.")
        
    except Exception as e:
        # 3. 로드 실패 시 (BitGenerator 오류 등), 이미 재저장된 FIXED 파일을 로드 시도
        st.warning(f"❌ 원본 모델 로드 실패: {e}. 재저장된 FIXED 파일로 대체합니다.")
        
        try:
            model = joblib.load(os.path.join(model_dir, 'classify_model_FIXED.pkl'))
            scaler = joblib.load(os.path.join(model_dir, 'scaler_FIXED.pkl'))
            st.info("✅ FIXED 파일 로드 성공.")
        except FileNotFoundError:
             st.error("❌ classify_model_FIXED.pkl 또는 scaler_FIXED.pkl 파일이 없습니다. 원본 파일을 확인해주세요.")
             return None, None, None, None
        except Exception as e_fixed:
             st.error(f"❌ FIXED 파일 로드도 실패: {e_fixed}")
             return None, None, None, None
             
    # 🚨🚨🚨 임시 재저장 및 로드 로직 끝 🚨🚨🚨
    
    # 나머지 에셋 로드 (오류 없이 진행되어야 합니다.)
    try:
        embedder = SentenceTransformer(EMBEDDER_MODEL_NAME) # SBERT 임베딩 모델
        with open(os.path.join(model_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
        return model, scaler, embedder, label_list
    except Exception as e:
        st.error(f"❌ 임베더/라벨 목록 로드 실패 (SBERT 라이브러리 문제일 수 있음): {e}")
        return None, None, None, None

@st.cache_resource
def load_summarization_model(model_path):
    # C 파트 요약 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device_sum)
    model.eval()
    return tokenizer, model



# ====================================================================
# B. 모델 로드 실행 및 추론 함수 정의
# ====================================================================

# 로드 실행
data_df = load_data(DATA_PATH)
model_clf, scaler_clf, embedder_clf, label_list_clf = load_classification_assets(CLASSIFY_MODEL_DIR)

# 요약 모델은 임시 비활성 상태 유지
try:
    tokenizer_sum, model_sum = load_summarization_model(MODEL_LOCAL_PATH)
    SUMMARY_LOADED = True
except Exception:
    SUMMARY_LOADED = False

# 1. 분류 추론 함수 (B 파트 로직)
def classify_petition(text):
    if not model_clf or not embedder_clf:
        return -1, "로드 실패 (에셋 확인 필요)"
        
    processed_text = text.strip() 
    
    # 2. SBERT 임베딩 (텍스트 -> 벡터 변환)
    text_vector = embedder_clf.encode([processed_text], convert_to_numpy=True)
    
    # 3. Scaler 적용 (데이터 정규화)
    text_scaled = scaler_clf.transform(text_vector)
    
    # 4. 예측
    prediction_index = model_clf.predict(text_scaled)[0]
    
    # 5. 예측된 인덱스 -> 카테고리 이름으로 변환
    try:
        category_code = label_list_clf[prediction_index] 
        category_name = FULL_CATEGORY_MAP.get(category_code, "분류 불가능")
        return category_code, category_name
    except IndexError:
        return -1, "매핑 오류 (IndexError)"


# 2. 요약 추론 함수 (C 파트 로직)
def summarize_petition(text, max_len=150):
    if not SUMMARY_LOADED:
        return "C 파트의 요약 모델 로드 실패로 기능을 사용할 수 없습니다."
        
    inputs = tokenizer_sum(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device_sum)


    with torch.no_grad():
        summary_ids = model_sum.generate(
            **inputs,
            max_length=max_len,
            min_length=40,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            length_penalty=1.0,
            early_stopping=True,
        )
    return tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)


# ====================================================================
# C. Streamlit 대시보드 화면 구성
# ====================================================================

st.title("🇰🇷 국회 청원 데이터 분석 및 AI 모델 대시보드")
st.markdown("### 📝 A파트: 최종 시스템 통합")

tab1, tab2, tab3 = st.tabs(["데이터 통계 및 시각화", "AI 모델 테스트 (분류)", "AI 모델 테스트 (요약)"])

with tab1:
    st.header("1. 데이터 통계 및 주요 현황")
    
    st.subheader("1-1. 카테고리별 청원 수")
    category_counts = data_df['category_name'].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']
    chart1 = alt.Chart(category_counts).mark_bar().encode(
        x=alt.X('Count', title='청원 수'),
        y=alt.Y('Category', sort='-x', title='청원 카테고리'),
        tooltip=['Category', 'Count']
    ).properties(title='카테고리별 청원 건수 분포').interactive()
    st.altair_chart(chart1, use_container_width=True)


    st.subheader("1-2. 동의 인원수 Top 10 청원")
    top_agree = data_df.nlargest(10, 'agree_count')
    st.dataframe(top_agree[['title', 'category_name', 'agree_count']], use_container_width=True)

    
    st.subheader("1-3. 동의 인원수 분포 (히스토그램)")
    st.caption("대부분의 청원이 낮은 동의수를 가지므로 로그 스케일로 변환하여 분포를 시각화합니다.")
    log_agree_count = data_df['agree_count'].apply(lambda x: np.log10(x + 1))
    chart2 = alt.Chart(pd.DataFrame({'log_agree_count': log_agree_count})).mark_bar().encode(
        alt.X("log_agree_count", bin=alt.Bin(maxbins=30), title="Log10(동의 인원 수 + 1)"),
        alt.Y("count()", title="청원 수"),
        tooltip=[alt.Tooltip("log_agree_count", bin=True), "count()"]
    ).properties(title='동의 인원수 분포 히스토그램').interactive()
    st.altair_chart(chart2, use_container_width=True)


with tab2:
    st.header("2. AI 모델 테스트 (분류)")
    st.markdown("---")
    
    st.subheader("2-1. 분류 모델 추론 자리")
    input_text = st.text_area("청원 내용을 입력하세요. (제목 없이 내용만)", height=300, key="input_text_classify")
    classify_button = st.button("분류 실행", key="classify_button", type="primary")
    
    st.markdown("#### 예측 결과:")
    
    if classify_button:
        if input_text:
            if model_clf and embedder_clf:
                with st.spinner('분류 중입니다...'):
                    pred_code, pred_name = classify_petition(input_text)
                    
                st.success("✅ 예측 완료")
                st.info(f"예측 카테고리 코드: **{pred_code}**")
                st.markdown(f"**최종 분류 결과:** ## {pred_name}")
            else:
                st.warning("분류 모델 로드에 문제가 있습니다. 에셋 파일 및 경로를 확인해주세요.")
        else:
            st.warning("분류할 내용을 입력해주세요.")


with tab3:
    st.header("3. AI 모델 테스트 (요약)")
    st.markdown("---")
    
    st.subheader("3-1. 요약 모델 추론 자리")
    petition_text = st.text_area(
        "요약할 청원 내용을 입력하세요.", 
        height=300, 
        key="input_text_summarize"
    )
    summarize_button = st.button("요약 실행", key="summarize_button", type="primary")
    
    st.markdown("#### 요약 결과:")
    
    if summarize_button:
        if petition_text:
            if SUMMARY_LOADED:
                with st.spinner('요약을 생성하는 중입니다...'):
                    final_summary = summarize_petition(petition_text)
                    
                st.success("✅ 요약 생성 완료")
                st.info(final_summary)
            else:
                st.warning("요약 모델 로드에 문제가 있습니다. 모델 파일 경로를 확인해주세요.")
        else:
            st.warning("요약할 내용을 입력해주세요.")