# --- 0. 필수 라이브러리 임포트 ---
import os
import pandas as pd
import numpy as np  # .npy 파일 로드를 위해 필요
import io
import json
import re  # 👈 정규식(Regex) 라이브러리 임포트
from typing import List, Set, TypedDict

# 머신러닝 및 모델 관련 라이브러리
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline # (기본 라이브러리 임포트 유지)
from google.cloud import vision
from google.oauth2 import service_account  # 👈 직접 인증을 위한 라이브러리

# LangGraph 라이브러리
from langgraph.graph import StateGraph, END

print("--- 🚀 알레르기 분석 서비스 (GCP Vision API + RAG + LLM Fallback) 시작 ---")
print("사전 빌드된 임베딩 캐시 파일을 로드합니다...")

# --- 0a. 표준 알레르기 목록 정의 ---
ALLERGENS_STD_SET = set([
    "알류", "우유", "메밀", "땅콩", "대두", "밀", "잣", "호두",
    "게", "새우", "오징어", "고등어", "조개류", "복숭아", "토마토",
    "닭고기", "돼지고기", "쇠고기", "아황산류"
])
print(f"✅ 표준 알레르기 카테고리 {len(ALLERGENS_STD_SET)}개 로드 완료.")

# --- 0b. 비-성분 키워드 필터 목록 (Node 2 수정용) ---
IGNORE_KEYWORDS = set([
    '열량', '탄수화물', '단백질', '지방', '당류', '나트륨', '콜레스테롤',
    '포화지방', '트랜스지방', '내용량', 'I', 'II' # (빈 문자열 '' 제거된 상태)
])
print(f"✅ 비-성분 필터 키워드 {len(IGNORE_KEYWORDS)}개 로드 완료.")


# --- 1. 글로벌 설정: 모델 로드 및 RAG 지식 베이스 캐시 로드 ---
try:
    # 1a. RAG 검색을 위한 임베딩 모델 로드 (쿼리 임베딩용)
    EMBEDDING_MODEL_NAME = 'distiluse-base-multilingual-cased-v1'
    print(f"'{EMBEDDING_MODEL_NAME}' 쿼리 임베딩 모델 로드 중...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ 쿼리 임베딩 모델 로드 완료.")

    # 1b. Zero-Shot NLI 모델 로드 (Fallback 전용) - [T5에서 교체됨]
    print("Zero-Shot NLI 모델 로드 중 (Fallback 전용)...")
    NLI_MODEL_NAME = "klue/roberta-base"
    
    # 이전 로그에서 CUDA 사용이 확인되었으므로 device=0 (GPU) 설정
    nli_pipeline = pipeline("zero-shot-classification", model=NLI_MODEL_NAME, device=0) 
    print(f"✅ Zero-Shot NLI 모델 ({NLI_MODEL_NAME}) 로드 완료.")
    
    # NLI Fallback이 사용할 후보 레이블 목록 (글로벌 캐시)
    ALLERGEN_CANDIDATES = list(ALLERGENS_STD_SET) + ["관련 없음"]
    print(f"✅ NLI Fallback 후보 레이블 {len(ALLERGEN_CANDIDATES)}개 준비 완료.")

    # 1c. GCP Vision API 클라이언트 초기화 (직접 경로 지정 방식)
    print("GCP Vision API 클라이언트 (직접 경로 지정) 초기화 중...")
    
    KEY_JSON_PATH = r"C:\Users\정주환\Desktop\keyfolder\nlp-study-467306-563e76afdbca.json"

    try:
        credentials = service_account.Credentials.from_service_account_file(KEY_JSON_PATH)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        print(f"✅ GCP Vision 클라이언트 (직접 경로) 준비 완료.")
        
    except FileNotFoundError:
        print(f"❌ 치명적 오류: 지정된 키 파일 경로를 찾을 수 없습니다! 경로를 다시 확인하세요.")
        print(f"   시도한 경로: {KEY_JSON_PATH}")
        raise 
    
    # 1d. 사전 계산된 벡터 캐시 로드
    print("사전 계산된 RAG 지식 베이스 캐시 로드 중...")
    kb_embeddings = np.load("kb_embeddings.npy") 
    
    with open("kb_categories.json", "r", encoding="utf-8") as f:
            kb_categories = json.load(f) 
    
    print(f"✅ RAG 지식 베이스 캐시 로드 완료 ({len(kb_categories)}개 항목)")

except Exception as e:
    print(f"❌ 치명적 오류: 글로벌 설정 실패: {e}")
    exit()


# RAG 검색 시 사용할 유사도 임계값 (튜닝된 값으로 가정, 예: 0.85)
# (만약 0.9로 유지하고 NLI가 '유청단백분말'을 잡도록 하려면 0.9로 설정하세요)
RAG_CONFIDENCE_THRESHOLD = 0.85
print(f"ℹ️ RAG 신뢰도 임계값: {RAG_CONFIDENCE_THRESHOLD}")

# NLI Fallback 결과가 유효하다고 인정할 최소 점수
NLI_FALLBACK_THRESHOLD = 0.5  
print(f"ℹ️ NLI Fallback 신뢰도 임계값: {NLI_FALLBACK_THRESHOLD}")

# --- 2. LangGraph 상태 정의 ---
class AllergyGraphState(TypedDict):
    image_path: str
    raw_ocr_text: str
    ingredients_to_check: List[str]
    current_ingredient: str
    rag_result: dict
    final_allergens: Set[str]
    final_output_json: str


# --- 3. LangGraph 노드 함수 정의 ---

def call_gcp_vision_api(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 1 (Entry Point): GCP Vision API 호출
    """
    print(f"\n--- (Node 1: call_gcp_vision_api) ---")
    img_path = state['image_path']
    print(f"GCP Vision API 호출... (이미지: {img_path})")
    try:
        with io.open(img_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        
        response = vision_client.text_detection(image=image)
        if response.error.message:
            raise Exception(f"GCP API Error: {response.error.message}")

        raw_text = response.full_text_annotation.text
        print(f"✅ GCP OCR 성공. (추출된 텍스트 길이: {len(raw_text)})")
        return {**state, "raw_ocr_text": raw_text}
    
    except Exception as e:
        print(f"❌ GCP Vision API 처리 실패: {e}")
        return {**state, "raw_ocr_text": ""}


def parse_text_from_raw(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 2 (Regex 파서 노드)
    (startswith 필터 로직이 적용된 최종 수정 버전)
    """
    print(f"\n--- (Node 2: parse_text_from_raw) [Regex Parser] ---")
    raw_text = state['raw_ocr_text']
    if not raw_text or not raw_text.strip():
        print("ℹ️ OCR 텍스트가 비어있어 파싱을 건너뜁니다.")
        return {**state, "ingredients_to_check": [], "final_allergens": set()}

    clean_text = raw_text.replace("\n", " ")

    ingredient_queue = []
    found_allergens_set = set()

    match1 = re.search(r"원재료명[ :](.*?)(•|\||영양정보|영양성분|$)", clean_text)
    
    if match1:
        ingredient_blob = match1.group(1).strip()
        raw_ingredients_list = [item.strip() for item in ingredient_blob.split(',') if item.strip()]
        cleaned_ingredients_raw = [name.split('(')[0].strip() for name in raw_ingredients_list if name.strip()]
        
        cleaned_ingredients_filtered = []
        for item in cleaned_ingredients_raw:
            is_noise = False
            for keyword in IGNORE_KEYWORDS:
                if item.startswith(keyword):
                    is_noise = True
                    print(f"  -> 필터링됨: '{item}' (노이즈 키워드 '{keyword}'로 시작하므로 제외)")
                    break 
            
            if not is_noise:
                cleaned_ingredients_filtered.append(item)
        
        ingredient_queue.extend(cleaned_ingredients_filtered)
        print(f"✅ Regex 파서: '원재료명' 섹션에서 {len(cleaned_ingredients_filtered)}개 성분 추출: {cleaned_ingredients_filtered}")
    
    else:
        print("ℹ️ Regex 파서: '원재료명' 섹션을 찾지 못함.")

    match2 = re.search(r"•?\s*([\w,]+)\s+함유", clean_text)
    if match2:
        contains_blob = match2.group(1) 
        contains_list = [item.strip() for item in contains_blob.split(',') if item.strip()]
        print(f"✅ Regex 파서: '...함유' 섹션에서 {len(contains_list)}개 항목 추출: {contains_list}")
        
        for item in contains_list:
            if item not in IGNORE_KEYWORDS:
                ingredient_queue.append(item) 
            
            if item in ALLERGENS_STD_SET:
                print(f"  -> '{item}'은(는) 표준 알레르기이므로 final_set에 직접 추가.")
                found_allergens_set.add(item) 
    else:
        print("ℹ️ Regex 파서: '...함유' 섹션을 찾지 못함.")

    final_queue = sorted(list(set(ingredient_queue)))
    print(f"==> 최종 RAG 검사 큐 (중복제거, {len(final_queue)}개): {final_queue}")
    
    return {
        **state,
        "ingredients_to_check": final_queue,      
        "final_allergens": found_allergens_set 
    }


def prepare_next_ingredient(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 3 (루프 컨트롤러)
    """
    print(f"\n--- (Node 3: prepare_next_ingredient) ---")
    queue = state['ingredients_to_check']
    next_ingredient = queue.pop(0) 
    print(f"다음 검사 대상: '{next_ingredient}' (남은 항목: {len(queue)}개)")
    return {
        **state,
        "current_ingredient": next_ingredient,
        "ingredients_to_check": queue
    }

def rag_search(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 4 (핵심 RAG 검색 노드)
    """
    print(f"--- (Node 4: rag_search) ---")
    ingredient = state['current_ingredient']
    
    query_embedding = embedding_model.encode([ingredient])
    similarities = cosine_similarity(query_embedding, kb_embeddings)
    
    best_match_index = np.argmax(similarities[0])
    confidence_score = float(similarities[0][best_match_index])
    
    found_allergen = kb_categories[best_match_index] 
    
    print(f"RAG 검색: '{ingredient}' (유사도: {confidence_score:.4f}) -> 매핑: '{found_allergen}'")
    
    rag_result_data = {
        "confidence": confidence_score,
        "found_allergen": found_allergen
    }
    return {**state, "rag_result": rag_result_data}


# ==============================================================================
# === 💥 [교체된 노드 5] (Zero-Shot NLI 파이프라인 버전) 💥 ===
# ==============================================================================
def llm_fallback(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 5 (LLM Fallback 노드) - [NLI Zero-Shot 버전]
    
    RAG가 실패한 항목을 Zero-Shot Classification 파이프라인(NLI 모델)으로 넘깁니다.
    입력 성분을 모든 알레르기 후보 레이블과 비교하여 최고 점수(entailment)를 받은 항목을 반환합니다.
    """
    print(f"--- (Node 5: llm_fallback) [NLI Zero-Shot] ---")
    ingredient = state['current_ingredient']
    print(f"NLI Fallback: '{ingredient}' 분류 요청... (후보: {len(ALLERGEN_CANDIDATES)}개)")

    try:
        # NLI 파이프라인 호출 (글로벌 파이프라인 'nli_pipeline' 및 후보 리스트 'ALLERGEN_CANDIDATES' 재사용)
        response = nli_pipeline(ingredient, ALLERGEN_CANDIDATES) 
        
        # 가장 점수가 높은 레이블과 점수를 추출
        top_label = response['labels'][0]
        top_score = response['scores'][0]
        
        print(f"NLI 응답: Label='{top_label}', Score={top_score:.4f}")

        # 최고 점수 레이블이 표준 알레르기 목록(SET)에 있는지 확인
        if top_label in ALLERGENS_STD_SET: 
            # 해당 점수가 우리가 설정한 NLI 임계값(예: 0.5)보다 높은지 확인
            if top_score >= NLI_FALLBACK_THRESHOLD:
                 print(f"  -> 유효한 분류: '{top_label}' (Score: {top_score}, 임계값 {NLI_FALLBACK_THRESHOLD} 통과).")
                 return {**state, "rag_result": {"confidence": top_score, "found_allergen": top_label}}
            else:
                 # 알레르기이긴 하지만, 점수가 너무 낮아서 신뢰할 수 없음
                 print(f"  -> 점수가 낮음 ({top_score} < {NLI_FALLBACK_THRESHOLD}). '없음'으로 처리.")
                 return {**state, "rag_result": {"confidence": 1.0, "found_allergen": "없음"}}
        else:
            # 최고 점수 레이블이 "관련 없음"이거나, (혹시 모를) 다른 쓰레기 값인 경우
            print(f"  -> 최고 점수 레이블이 '{top_label}'이므로 '없음'으로 처리.")
            return {**state, "rag_result": {"confidence": 1.0, "found_allergen": "없음"}}
            
    except Exception as e:
        print(f"❌ NLI Fallback 중 오류: {e}")
        return {**state, "rag_result": {"confidence": 1.0, "found_allergen": "오류"}}
# ==============================================================================
# === [교체된 노드 5] 끝 ===
# ==============================================================================


def update_final_list(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 6 (결과 취합 노드)
    """
    print(f"--- (Node 6: update_final_list) ---")
    result_allergen = state['rag_result']['found_allergen']
    
    if result_allergen in ALLERGENS_STD_SET:
        current_set = state['final_allergens']
        print(f"✅ 유효한 알레르기 발견: '{result_allergen}'. 최종 목록에 추가.")
        current_set.add(result_allergen)
        return {**state, "final_allergens": current_set}
    else:
        print(f"ℹ️ '{result_allergen}'은(는) 표준 알레르기 항목이 아니므로 무시합니다.")
        return state 

def finalize_processing(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 7 (종료 노드)
    """
    print(f"\n--- (Node 7: finalize_processing) ---")
    final_set = state['final_allergens']
    
    final_list = sorted(list(final_set))
    final_json = json.dumps(final_list, ensure_ascii=False)
    
    print(f"🎉 모든 성분 검사 완료. 최종 결과: {final_json}")
    return {**state, "final_output_json": final_json}


# --- 4. LangGraph 엣지(Edge) 함수 정의 ---

def route_rag_result(state: AllergyGraphState) -> str:
    """(조건부 엣지 1: RAG 라우터)
    """
    print(f"--- (Edge: route_rag_result?) ---")
    confidence = state['rag_result']['confidence']
    allergen = state['rag_result']['found_allergen']
    
    if confidence >= RAG_CONFIDENCE_THRESHOLD and allergen in ALLERGENS_STD_SET:
        print(f"  -> [RAG 성공]. 'update_final_list'로 이동.")
        return "rag_success"
    else:
        print(f"  -> [RAG 실패/불확실]. 'llm_fallback'으로 이동.")
        return "needs_llm_fallback"

def check_remaining_ingredients(state: AllergyGraphState) -> str:
    """(조건부 엣지 2: 루프 제어)
    """
    print(f"--- (Edge: check_remaining_ingredients?) ---")
    
    if state["ingredients_to_check"] and len(state["ingredients_to_check"]) > 0:
        print(f"  -> [항목 남음]. 'prepare_next_ingredient'로 루프.")
        return "has_more_ingredients"
    else:
        print("  -> [항목 없음]. 'finalize_processing'로 이동.")
        return "all_ingredients_done"

# --- 5. 그래프 빌드 및 컴파일 ---

print("\n--- LangGraph 워크플로우 빌드 시작 ---")

workflow = StateGraph(AllergyGraphState)

# 1. 모든 노드를 그래프에 추가
workflow.add_node("call_gcp_vision_api", call_gcp_vision_api)
workflow.add_node("parse_text_from_raw", parse_text_from_raw)
workflow.add_node("prepare_next_ingredient", prepare_next_ingredient)
workflow.add_node("rag_search", rag_search)
workflow.add_node("llm_fallback", llm_fallback) # <--- NLI 버전 함수가 등록됨
workflow.add_node("update_final_list", update_final_list)
workflow.add_node("finalize_processing", finalize_processing)

# 2. 엣지 연결 (흐름 정의)
workflow.set_entry_point("call_gcp_vision_api")                
workflow.add_edge("call_gcp_vision_api", "parse_text_from_raw")    
workflow.add_edge("parse_text_from_raw", "prepare_next_ingredient") 
workflow.add_edge("prepare_next_ingredient", "rag_search")          

# 3. RAG 라우팅 (조건부 엣지 1)
workflow.add_conditional_edges(
    "rag_search",
    route_rag_result,
    {"rag_success": "update_final_list", "needs_llm_fallback": "llm_fallback"}
)

# 4. Fallback 결과도 취합 노드로 연결 (NLI Fallback이 연결됨)
workflow.add_edge("llm_fallback", "update_final_list") 

# 5. 메인 루프 (조건부 엣지 2)
workflow.add_conditional_edges(
    "update_final_list",
    check_remaining_ingredients,
    {"has_more_ingredients": "prepare_next_ingredient", "all_ingredients_done": "finalize_processing"}
)

# 6. 종료 노드 연결
workflow.add_edge("finalize_processing", END)

# 7. 컴파일
app = workflow.compile()
print("--- ✅ LangGraph 워크플로우 컴파일 완료 ---")


# --- 6. 테스트 실행 ---
print("\n\n--- [Test Run: GCP API + Regex 파서 + NLI Fallback 기반 실행] ---")

# (테스트할 이미지 파일을 지정해야 합니다)
my_test_image_file = "image.jpg" # 👈 'image.jpg'는 OCR 로그를 제공한 그 이미지 파일 가정

if my_test_image_file:
    test_input = {"image_path": my_test_image_file}
    print(f"테스트 실행 시작: {my_test_image_file}\n")

    print("\n--- [Test Run: 최종 결과 (invoke)] ---")
    final_state = app.invoke(test_input, {"recursion_limit": 100}) 
    print("\n최종 반환 JSON:")
    print(final_state['final_output_json'])

else:
    print("\n테스트 실행 건너뜀: 'my_test_image_file' 변수에 이미지 경로가 지정되지 않았습니다.")