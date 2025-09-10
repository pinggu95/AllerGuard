import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import time

print("--- [Phase 1: 임베딩 캐시 빌더] 시작 ---")
print("이 스크립트는 CSV 데이터를 읽어 임베딩을 계산하고 파일로 저장합니다.")
print("시간이 다소 걸릴 수 있습니다...\n")

try:
    # 1. CSV 데이터 로드
    file_path = "domestic_allergy_rag_knowledge_1000.csv"
    df = pd.read_csv(file_path).dropna(subset=['term', 'category'])
    
    # RAG 키(term)와 값(category) 추출
    terms_list = df['term'].tolist()
    categories_list = df['category'].tolist()
    
    print(f"✅ CSV 로드 완료. {len(terms_list)}개의 지식 데이터 확인.")

    # 2. 임베딩 모델 로드 (데이터 생성용)
    model_name = 'distiluse-base-multilingual-cased-v1'
    print(f"'{model_name}' 임베딩 모델 로드 중...")
    start_time = time.time()
    model = SentenceTransformer(model_name)
    print(f"✅ 모델 로드 완료. (소요 시간: {time.time() - start_time:.2f}초)")

    # 3. 'term' 컬럼 전체 임베딩 생성
    print(f"\n{len(terms_list)}개의 'term'에 대한 임베딩 생성 시작...")
    start_time = time.time()
    term_embeddings = model.encode(terms_list, convert_to_tensor=False, show_progress_bar=True)
    
    # 4. (오류 수정) Numpy 배열로 변환
    # ❌ (기존 오류 코드): .astype('np.float32')
    # ✅ (수정된 코드): .astype(np.float32) <- 따옴표 제거
    term_embeddings_np = np.array(term_embeddings).astype(np.float32)
    
    print(f"\n✅ 임베딩 생성 및 Numpy 변환 완료. (소요 시간: {time.time() - start_time:.2f}초)")
    
    # --- 5. 계산된 결과(캐시)를 파일로 저장 ---

    # 5a. 임베딩 벡터 배열을 Numpy 바이너리 파일(.npy)로 저장
    cache_file_vectors = "kb_embeddings.npy"
    np.save(cache_file_vectors, term_embeddings_np)
    print(f"💾 벡터 캐시가 '{cache_file_vectors}' 파일로 저장되었습니다.")
    
    # 5b. 인덱스-카테고리 매핑 JSON 파일 저장
    cache_file_categories = "kb_categories.json"
    with open(cache_file_categories, "w", encoding="utf-8") as f:
        json.dump(categories_list, f, ensure_ascii=False, indent=2)
    print(f"💾 카테고리 매핑이 '{cache_file_categories}' 파일로 저장되었습니다.")
    
    print("\n--- ✅ [Phase 1: 임베딩 캐시 빌드] 성공 ---")

except Exception as e:
    print(f"❌ 처리 중 심각한 오류 발생: {e}")