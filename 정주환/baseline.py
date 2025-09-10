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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
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
# 파서가 실수로 추출한 영양정보 또는 OCR 노이즈를 RAG 큐에서 제외하기 위한 필터
IGNORE_KEYWORDS = set([
    '열량', '탄수화물', '단백질', '지방', '당류', '나트륨', '콜레스테롤',
    '포화지방', '트랜스지방', '내용량', 'I', 'II' # <-- 빈 문자열 '' 제거됨
])
print(f"✅ 비-성분 필터 키워드 {len(IGNORE_KEYWORDS)}개 로드 완료.")


# --- 1. 글로벌 설정: 모델 로드 및 RAG 지식 베이스 캐시 로드 ---
# (앱 실행 시 단 1회 수행. 모든 모델과 데이터를 메모리에 로드합니다.)
try:
    # 1a. RAG 검색을 위한 임베딩 모델 로드 (쿼리 임베딩용)
    EMBEDDING_MODEL_NAME = 'distiluse-base-multilingual-cased-v1'
    print(f"'{EMBEDDING_MODEL_NAME}' 쿼리 임베딩 모델 로드 중...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ 쿼리 임베딩 모델 로드 완료.")

    # 1b. T5 모델 로드 (Fallback 용도로만 사용됨)
    print("T5 모델 로드 중 (Fallback 전용)...")
    t5_model_id = "paust/pko-t5-small"
    t5_tok = AutoTokenizer.from_pretrained(t5_model_id)
    t5_mdl = AutoModelForSeq2SeqLM.from_pretrained(t5_model_id)
    t5_pipeline = pipeline("text2text-generation", model=t5_mdl, tokenizer=t5_tok, max_new_tokens=64)
    print("✅ T5 모델 로드 완료.")

    # 1c. GCP Vision API 클라이언트 초기화 (직접 경로 지정 방식)
    print("GCP Vision API 클라이언트 (직접 경로 지정) 초기화 중...")
    
    # 사용자님의 PC에 있는 키 파일의 절대 경로 (한글 경로 포함)
    KEY_JSON_PATH = r"C:\Users\정주환\Desktop\keyfolder\nlp-study-467306-563e76afdbca.json"

    try:
        # 1. 지정된 경로에서 서비스 계정 파일로 '인증정보(credentials)' 객체를 직접 생성
        credentials = service_account.Credentials.from_service_account_file(KEY_JSON_PATH)
        
        # 2. 클라이언트 생성 시 이 '인증정보'를 수동으로 주입(inject)
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        
        print(f"✅ GCP Vision 클라이언트 (직접 경로) 준비 완료.")
        
    except FileNotFoundError:
        print(f"❌ 치명적 오류: 지정된 키 파일 경로를 찾을 수 없습니다! 경로를 다시 확인하세요.")
        print(f"   시도한 경로: {KEY_JSON_PATH}")
        raise # 파일이 없으면 앱 중지
    
    # 1d. 사전 계산된 벡터 캐시 로드
    print("사전 계산된 RAG 지식 베이스 캐시 로드 중...")
    kb_embeddings = np.load("kb_embeddings.npy") # Numpy 배열을 파일에서 바로 로드
    
    with open("kb_categories.json", "r", encoding="utf-8") as f:
            kb_categories = json.load(f) # 카테고리 매핑 리스트를 JSON에서 로드
    
    print(f"✅ RAG 지식 베이스 캐시 로드 완료 ({len(kb_categories)}개 항목)")

except Exception as e:
    print(f"❌ 치명적 오류: 글로벌 설정 실패: {e}")
    # 설정이 실패하면 앱을 실행할 수 없으므로 종료
    exit()


# RAG 검색 시 사용할 유사도 임계값
RAG_CONFIDENCE_THRESHOLD = 0.85

# --- 2. LangGraph 상태 정의 ---
class AllergyGraphState(TypedDict):
    """
    그래프 전체를 순회하는 중앙 상태 저장소(State)입니다.
    모든 노드는 이 State를 읽고, 자신의 작업 결과를 이 State에 다시 씁니다.
    """
    image_path: str                # 그래프 최초 입력 (이미지 경로)
    raw_ocr_text: str              # GCP OCR이 반환한 원본 텍스트
    ingredients_to_check: List[str]  # 파싱된 후, 검사를 기다리는 성분 목록 (큐)
    current_ingredient: str        # 현재 루프에서 검사 중인 단일 성분
    rag_result: dict               # RAG 또는 LLM 노드의 처리 결과
    final_allergens: Set[str]      # 최종 발견된 표준 알레르기 (중복 제거용 Set)
    final_output_json: str         # 사용자에게 반환될 최종 JSON 문자열


# --- 3. LangGraph 노드 함수 정의 ---

def call_gcp_vision_api(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 1 (Entry Point): GCP Vision API 호출
    
    State의 'image_path'를 받아 GCP Vision API를 호출하고,
    추출된 전체 텍스트 블록을 'raw_ocr_text' 상태로 업데이트합니다.
    """
    print(f"\n--- (Node 1: call_gcp_vision_api) ---")
    img_path = state['image_path']
    print(f"GCP Vision API 호출... (이미지: {img_path})")
    try:
        # 이미지 파일을 바이너리로 읽기
        with io.open(img_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        
        # 텍스트 감지(text_detection) API 호출
        response = vision_client.text_detection(image=image)
        if response.error.message:
            raise Exception(f"GCP API Error: {response.error.message}")

        # 모든 텍스트를 하나의 문자열 블록으로 가져옴
        raw_text = response.full_text_annotation.text
        print(f"✅ GCP OCR 성공. (추출된 텍스트 길이: {len(raw_text)})")
        return {**state, "raw_ocr_text": raw_text}
    
    except Exception as e:
        print(f"❌ GCP Vision API 처리 실패: {e}")
        return {**state, "raw_ocr_text": ""}


def parse_text_from_raw(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 2 (Regex 파서 노드)
    
    'raw_ocr_text'를 입력받아 정규식(Regex)으로 파싱합니다.
    1. '원재료명:' 섹션에서 성분 목록을 추출합니다. (예: '밀가루', '치즈분말')
    2. '...함유' 섹션에서 경고 목록을 추출합니다. (예: '밀', '우유')
    
    '함유' 목록의 표준 알레르기는 'final_allergens' Set에 미리 추가하고,
    두 목록의 모든 성분(중복 제거)을 'ingredients_to_check' 큐로 반환합니다.
    """
    print(f"\n--- (Node 2: parse_text_from_raw) [Regex Parser] ---")
    raw_text = state['raw_ocr_text']
    if not raw_text or not raw_text.strip():
        print("ℹ️ OCR 텍스트가 비어있어 파싱을 건너뜁니다.")
        return {**state, "ingredients_to_check": [], "final_allergens": set()}

    # 정규식 처리를 쉽게 하기 위해 모든 개행문자(\n)를 공백으로 치환
    clean_text = raw_text.replace("\n", " ")

    ingredient_queue = []      # RAG가 검사할 모든 성분 후보 큐
    found_allergens_set = set() # 최종 결과를 누적할 Set 초기화

    # 1. "원재료명:" 섹션 추출 (영양정보 섹션 전까지만 읽도록 Regex 수정됨)
    match1 = re.search(r"원재료명[ :](.*?)(•|\||영양정보|영양성분|$)", clean_text)
    
    if match1:
        ingredient_blob = match1.group(1).strip() # "밀가루(밀:미국산), 가공유지..."
        raw_ingredients_list = [item.strip() for item in ingredient_blob.split(',') if item.strip()]
        
        # 성분 이름만 정리 (예: "밀가루(밀:미국산)" -> "밀가루")
        cleaned_ingredients_raw = [name.split('(')[0].strip() for name in raw_ingredients_list if name.strip()]
        
        # 'startswith' 필터링 로직 (노이즈 청크 제거)
        cleaned_ingredients_filtered = []
        for item in cleaned_ingredients_raw:
            is_noise = False
            for keyword in IGNORE_KEYWORDS:
                if item.startswith(keyword):  # IGNORE_KEYWORDS 키워드로 시작하면
                    is_noise = True
                    print(f"  -> 필터링됨: '{item}' (노이즈 키워드 '{keyword}'로 시작하므로 제외)")
                    break  # 노이즈 확인 시 내부 루프 탈출
            
            if not is_noise:
                cleaned_ingredients_filtered.append(item) # 노이즈가 아닌 항목만 추가
        
        ingredient_queue.extend(cleaned_ingredients_filtered)
        print(f"✅ Regex 파서: '원재료명' 섹션에서 {len(cleaned_ingredients_filtered)}개 성분 추출: {cleaned_ingredients_filtered}")
    
    else:
        print("ℹ️ Regex 파서: '원재료명' 섹션을 찾지 못함.")

    # 2. "... 함유" 섹션에서 모든 알레르기 직접 추출
    match2 = re.search(r"•?\s*([\w,]+)\s+함유", clean_text)
    if match2:
        contains_blob = match2.group(1) # "밀,우유,대두,쇠고기"
        contains_list = [item.strip() for item in contains_blob.split(',') if item.strip()]
        print(f"✅ Regex 파서: '...함유' 섹션에서 {len(contains_list)}개 항목 추출: {contains_list}")
        
        for item in contains_list:
            if item not in IGNORE_KEYWORDS: # (함유 목록에도 안전 필터 적용)
                ingredient_queue.append(item) 
            
            if item in ALLERGENS_STD_SET: # 표준 알레르기 목록에 있다면
                print(f"  -> '{item}'은(는) 표준 알레르기이므로 final_set에 직접 추가.")
                found_allergens_set.add(item) # 최종 목록에 미리 추가
    else:
        print("ℹ️ Regex 파서: '...함유' 섹션을 찾지 못함.")

    # 3. 최종 큐 생성 (중복 제거)
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
    
    'ingredients_to_check' 큐(Queue)에서 성분을 하나씩 꺼내어(pop)
    'current_ingredient' 상태에 설정합니다.
    """
    print(f"\n--- (Node 3: prepare_next_ingredient) ---")
    queue = state['ingredients_to_check']
    next_ingredient = queue.pop(0) # 큐의 맨 앞에서 하나 꺼냄
    print(f"다음 검사 대상: '{next_ingredient}' (남은 항목: {len(queue)}개)")
    return {
        **state,
        "current_ingredient": next_ingredient, # 현재 검사할 대상 설정
        "ingredients_to_check": queue          # 하나가 제거된 큐로 상태 업데이트
    }

def rag_search(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 4 (핵심 RAG 검색 노드)
    
    'current_ingredient'를 임베딩하고, 메모리에 로드된 KB 벡터 전체와 비교합니다.
    가장 유사도(confidence)가 높은 항목의 매핑된 알레르기 값을 'rag_result' 상태로 반환합니다.
    """
    print(f"--- (Node 4: rag_search) ---")
    ingredient = state['current_ingredient']
    
    # 1. 쿼리(성분 1개) 임베딩 생성 (실시간)
    query_embedding = embedding_model.encode([ingredient])
    
    # 2. KB(702개) 벡터 전체와 코사인 유사도 계산
    similarities = cosine_similarity(query_embedding, kb_embeddings)
    
    # 3. 최고 점수(argmax)의 인덱스 탐색
    best_match_index = np.argmax(similarities[0])
    confidence_score = float(similarities[0][best_match_index])
    
    # 4. 해당 인덱스의 알레르기 값 매핑 (kb_categories 리스트에서 조회)
    found_allergen = kb_categories[best_match_index] 
    
    print(f"RAG 검색: '{ingredient}' (유사도: {confidence_score:.4f}) -> 매핑: '{found_allergen}'")
    
    rag_result_data = {
        "confidence": confidence_score,
        "found_allergen": found_allergen
    }
    return {**state, "rag_result": rag_result_data}

def llm_fallback(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 5 (LLM Fallback 노드)
    
    RAG 검색 결과가 불확실할 때(신뢰도 임계값 미만) 호출됩니다.
    글로벌 T5 모델에게 '분류' 질문을 던져 이 성분이 어떤 표준 알레르기인지 교차 검증합니다.
    """
    print(f"--- (Node 5: llm_fallback) ---")
    ingredient = state['current_ingredient']
    print(f"LLM Fallback: T5 모델에게 '{ingredient}' 분류 요청...")

    # T5 모델을 '분류기'로 활용하는 프롬프트
    prompt = f"""
다음 성분은 어떤 표준 알레르기 분류에 속합니까? 성분: "{ingredient}", 분류 목록: {', '.join(list(ALLERGENS_STD_SET))}. 지시: 목록 중 하나만 정확히 답변하세요. 목록에 없으면 "없음"이라고만 답변하세요.
정답: """
    
    try:
        # 글로벌 T5 파이프라인 재사용
        response = t5_pipeline(prompt)[0]["generated_text"].strip()
        print(f"T5 응답: '{response}'")
        
        if response in ALLERGENS_STD_SET: # T5 응답이 표준 알레르기 목록에 있다면
            return {**state, "rag_result": {"confidence": 1.0, "found_allergen": response}}
        else: # "없음" 또는 기타 쓰레기 값을 반환하면 "없음"으로 통일
            return {**state, "rag_result": {"confidence": 1.0, "found_allergen": "없음"}}
            
    except Exception as e:
        print(f"❌ T5 Fallback 중 오류: {e}")
        return {**state, "rag_result": {"confidence": 1.0, "found_allergen": "오류"}}

def update_final_list(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 6 (결과 취합 노드)
    
    RAG(노드 4) 또는 LLM(노드 5)의 처리 결과를 받아,
    그 결과가 '표준 알레르기 목록(ALLERGENS_STD_SET)'에 포함된 유효한 항목일 경우에만
    'final_allergens' Set에 추가(누적)합니다.
    """
    print(f"--- (Node 6: update_final_list) ---")
    result_allergen = state['rag_result']['found_allergen']
    
    if result_allergen in ALLERGENS_STD_SET:
        current_set = state['final_allergens']
        # '함유' 목록(노드 2)에서 이미 추가되었을 수도 있지만, Set이므로 중복은 자동 처리됨
        print(f"✅ 유효한 알레르기 발견: '{result_allergen}'. 최종 목록에 추가.")
        current_set.add(result_allergen)
        return {**state, "final_allergens": current_set}
    else:
        # 결과가 "없음", "오류" 또는 KB의 "기타" 등 표준 목록에 없으면 무시
        print(f"ℹ️ '{result_allergen}'은(는) 표준 알레르기 항목이 아니므로 무시합니다.")
        return state # Set에 변경이 없으므로 state를 그대로 반환

def finalize_processing(state: AllergyGraphState) -> AllergyGraphState:
    """
    ✅ 노드 7 (종료 노드)
    
    모든 루프가 끝난 후 호출됩니다.
    최종 누적된 'final_allergens' Set을 API 응답에 적합한 '정렬된 JSON 리스트'로 변환합니다.
    """
    print(f"\n--- (Node 7: finalize_processing) ---")
    final_set = state['final_allergens']
    
    # Set(순서 없음)을 List로 변환하고 알파벳순으로 정렬
    final_list = sorted(list(final_set))
    
    # JSON 문자열로 변환
    final_json = json.dumps(final_list, ensure_ascii=False)
    
    print(f"🎉 모든 성분 검사 완료. 최종 결과: {final_json}")
    return {**state, "final_output_json": final_json}


# --- 4. LangGraph 엣지(Edge) 함수 정의 ---

def route_rag_result(state: AllergyGraphState) -> str:
    """(조건부 엣지 1: RAG 라우터)
    RAG 검색(노드 4)의 신뢰도를 확인하여, 다음 단계를 결정합니다.
    - [성공] 신뢰도가 높고 유효한 알레르기 -> 'update_final_list'로 바로 이동
    - [실패/불확실] 신뢰도가 낮거나 유효하지 않은 값 -> 'llm_fallback'으로 이동
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
    결과 취합(노드 6) 후, 'ingredients_to_check' 큐에 검사할 항목이 더 남아있는지 확인합니다.
    - [남음] 큐에 항목이 남아있음 -> 'prepare_next_ingredient'로 돌아가 루프 계속
    - [없음] 큐가 비었음 -> 'finalize_processing'로 이동하여 그래프 종료
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
workflow.add_node("parse_text_from_raw", parse_text_from_raw) # <--- 최종 수정된 함수가 등록됨
workflow.add_node("prepare_next_ingredient", prepare_next_ingredient)
workflow.add_node("rag_search", rag_search)
workflow.add_node("llm_fallback", llm_fallback)
workflow.add_node("update_final_list", update_final_list)
workflow.add_node("finalize_processing", finalize_processing)

# 2. 엣지 연결 (흐름 정의)
workflow.set_entry_point("call_gcp_vision_api")                # 시작: GCP 호출
workflow.add_edge("call_gcp_vision_api", "parse_text_from_raw")    # GCP -> 파싱 (수정된 노드)
workflow.add_edge("parse_text_from_raw", "prepare_next_ingredient") # 파싱 -> 루프 시작(첫 성분 준비)
workflow.add_edge("prepare_next_ingredient", "rag_search")          # 성분 준비 -> RAG 검색

# 3. RAG 라우팅 (조건부 엣지 1)
workflow.add_conditional_edges(
    "rag_search",
    route_rag_result,
    {"rag_success": "update_final_list", "needs_llm_fallback": "llm_fallback"}
)

# 4. Fallback 결과도 취합 노드로 연결
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
print("\n\n--- [Test Run: GCP API + Regex 파서 + 로컬 캐시 기반 실행] ---")

# (테스트할 이미지 파일을 지정해야 합니다)
my_test_image_file = "image.jpg" # 👈 'image.jpg'는 OCR 로그를 제공한 그 이미지 파일 가정

if my_test_image_file:
    test_input = {"image_path": my_test_image_file}
    print(f"테스트 실행 시작: {my_test_image_file}\n")

    # (주석 해제하여 스트림 로그 보기)
    # for step in app.stream(test_input, {"recursion_limit": 50}): 
    #     print(step)

    print("\n--- [Test Run: 최종 결과 (invoke)] ---")
    # .invoke()는 모든 단계를 실행하고 최종 상태(State)만 반환함
    # 큐의 아이템 개수가 30개를 초과할 수 있으므로, 재귀 제한을 넉넉하게 100으로 설정
    final_state = app.invoke(test_input, {"recursion_limit": 100}) 
    print("\n최종 반환 JSON:")
    print(final_state['final_output_json'])

else:
    print("\n테스트 실행 건너뜀: 'my_test_image_file' 변수에 이미지 경로가 지정되지 않았습니다.")