# -*- coding: utf-8 -*-

import os
import io
import json
import re
from typing import List, Set, TypedDict

import numpy as np
import torch
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # (호환성 유지용; 직접 dot 사용)
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as hf_logging

# GCP Vision OCR
from google.cloud import vision
from google.oauth2 import service_account

# (선택) Gemini Structured Output
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False

# (선택) Document AI
try:
    from google.cloud import documentai
    _HAS_DOCAI = True
except Exception:
    _HAS_DOCAI = False

# LangGraph
from langgraph.graph import StateGraph, END

print("--- 🚀 알레르기 분석 서비스 (GCP Vision API + RAG + LLM Fallback) 시작 ---")

# =====================
# 0. 전역 설정/상수
# =====================
ALLERGENS_STD_SET = set([
    "알류", "우유", "메밀", "땅콩", "대두", "밀", "잣", "호두",
    "게", "새우", "오징어", "고등어", "조개류", "복숭아", "토마토",
    "닭고기", "돼지고기", "쇠고기", "아황산류"
])
print(f"✅ 표준 알레르기 카테고리 {len(ALLERGENS_STD_SET)}개 로드 완료.")

IGNORE_KEYWORDS = set([
    "열량", "탄수화물", "단백질", "지방", "당류", "나트륨", "콜레스테롤",
    "포화지방", "트랜스지방", "내용량", "I", "II"
])
print(f"✅ 비-성분 필터 키워드 {len(IGNORE_KEYWORDS)}개 로드 완료.")

# 동의어→표준 매핑
ALIAS2STD = {
    # 알류(난류)
    "난류": "알류", "계란": "알류", "달걀": "알류", "난백": "알류", "난황": "알류",
    # 우유 계열
    "유청": "우유", "유청단백": "우유", "유청단백분말": "우유", "카제인": "우유", "카제인나트륨": "우유",
    "치즈": "우유", "치즈분말": "우유", "탈지분유": "우유", "분유": "우유",
    # 대두/밀/견과
    "대두레시틴": "대두", "레시틴(대두)": "대두", "밀가루": "밀", "땅콩버터": "땅콩",
    "호두분태": "호두", "잣가루": "잣",
    # 수산물/조개류
    "홍합": "조개류", "굴": "조개류", "전복": "조개류",
    "고등어추출물": "고등어", "새우추출물": "새우", "오징어먹물": "오징어",
    # 과채, 첨가물
    "복숭아농축액": "복숭아", "토마토페이스트": "토마토",
    "아황산나트륨": "아황산류",
}

# 경로/모델 설정(환경변수 우선, 없으면 존재 여부에 따라 자동 선택)
KB_EMB_PATH = r"C:\\Users\\MYNOTE\\AllerGuard\\차지예\\kb_embeddings.npy"
KB_CAT_PATH = r"C:\\Users\\MYNOTE\\AllerGuard\\차지예\\kb_categories.json"
KB_CSV_PATH = r"C:\\Users\\MYNOTE\\AllerGuard\\domestic_allergy_rag_knowledge_1000.csv"


# GCP Vision Key
KEY_JSON_PATH = os.environ.get("GCP_VISION_KEY_PATH", r"D:\key folder\ocr-project-470906-7ffeebabeb09.json")

EMBEDDING_MODEL_NAME = "distiluse-base-multilingual-cased-v1"
NLI_MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

# 파서 선택: "gemini"(기본) 또는 "docai"
USE_API_PARSER = os.environ.get("ALLER_GUARD_API_PARSER", "gemini").lower()

# 임계값
RAG_CONFIDENCE_THRESHOLD = float(os.environ.get("RAG_CONF_THRESH", 0.85))
NLI_FALLBACK_THRESHOLD   = float(os.environ.get("NLI_FALLBACK_THRESH", 0.5))

print(f"ℹ️ RAG 임계값={RAG_CONFIDENCE_THRESHOLD}, NLI 임계값={NLI_FALLBACK_THRESHOLD}")
print(f"ℹ️ API 파서 모드: {USE_API_PARSER}")

# --- Gemini API 키 탐색 도우미 ---
HARDCODED_GEMINI_API_KEY = "AIzaSyDMTVeVGPU374hlJWEGhxB902f-RxkRVSU"  # ❗ 보안상 빈 문자열 유지. 키는 환경변수로 넣으세요.

def _get_gemini_api_key():
    # 0) 코드에 직접 입력된 키 (권장X)
    if HARDCODED_GEMINI_API_KEY:
        return HARDCODED_GEMINI_API_KEY.strip()
    # 1) 대표 환경변수 이름들 시도
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    # 2) .env 지원 (선택)
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"):
            v = os.environ.get(var)
            if v:
                return v
    except Exception:
        pass
    # 3) 로컬 키 파일 (선택)
    for fname in ("gemini_api_key.txt", ".gemini_api_key"):
        if os.path.exists(fname):
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                    if key:
                        return key
            except Exception:
                pass
    return None

# =====================
# 1. 유틸 (정규화/용어-핵심 가드)
# =====================
GENERIC_SUFFIXES = (
    "가루","분말","추출물","농축액","농축분말","유래","단백질","농축",
    "페이스트","엑기스","분태","시럽","오일","혼합","액","분","정제","가수분해물"
)

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def normalize_to_std(name: str) -> str:
    n = re.sub(r"\s+", "", str(name))
    n = n.split("(")[0]
    return ALIAS2STD.get(n, n)


def core_token(s: str) -> str:
    s = re.sub(r"\s+", "", str(s))
    s = s.split("(")[0]
    # 뒤에서부터 한 번만 제거 (과잉 제거 방지)
    for suf in GENERIC_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[:-len(suf)]
            break
    return s


def lexical_consistent(query: str, cand_term: str) -> bool:
    q = core_token(query)
    c = core_token(cand_term)
    if not q or not c:
        return False
    if q == c:
        return True
    # 2글자 이상 핵심어의 포함 관계면 유사하다고 간주
    if len(q) >= 2 and len(c) >= 2 and (q in c or c in q):
        return True
    return False

# =====================
# 2. 글로벌 리소스 초기화
# =====================
try:
    print(f"'{EMBEDDING_MODEL_NAME}' 쿼리 임베딩 모델 로드 중...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ 쿼리 임베딩 모델 로드 완료.")

    print("Zero-Shot NLI 모델 로드 중 (Fallback 전용)...")
    hf_logging.set_verbosity_error()
    try:
        import sentencepiece  # noqa: F401
    except Exception:
        print("⚠️ 'sentencepiece' 패키지가 없습니다. 'pip install sentencepiece' 권장(멀티링구얼 모델에 필요)")

    # 안전한 NLI 로더(순차 폴백)
    candidates = [
        ("joeddav/xlm-roberta-large-xnli", False),
        ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", False),
        ("facebook/bart-large-mnli", True),  # 영어 전용(긴급 폴백)
    ]
    last_err = None
    nli_pipeline = None
    for mid, english_only in candidates:
        try:
            nli_tokenizer = AutoTokenizer.from_pretrained(mid, use_fast=False)
            nli_model = AutoModelForSequenceClassification.from_pretrained(mid)
            nli_pipeline = pipeline(
                "zero-shot-classification",
                model=nli_model,
                tokenizer=nli_tokenizer,
                device=(0 if torch.cuda.is_available() else -1),
                hypothesis_template=(
                    "이 성분은 {} 알레르겐(과)에 해당한다." if not english_only else "This ingredient belongs to {} allergen."
                ),
            )
            NLI_MODEL_NAME = mid
            print(f"✅ NLI 모델 로드: {mid}")
            break
        except Exception as e:
            print(f"⚠️ NLI 후보 로드 실패({mid}): {e}")
            last_err = e
    if nli_pipeline is None:
        raise RuntimeError(f"NLI 모델 로드 실패(모든 후보 실패): {last_err}")

    # NLI 후보 레이블
    ALLERGEN_CANDIDATES = list(ALLERGENS_STD_SET) + ["관련 없음"]

    # GCP Vision 클라이언트
    print("GCP Vision API 클라이언트 초기화 중...")
    credentials = service_account.Credentials.from_service_account_file(KEY_JSON_PATH)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    print("✅ GCP Vision 클라이언트 준비 완료.")

    # KB 로드 + L2 정규화 + 중복 제거 + 텍스트/용어 매핑
    print("사전 계산된 RAG 지식 베이스 로드 중...")

    if not os.path.exists(KB_EMB_PATH) or not os.path.exists(KB_CAT_PATH):
        raise FileNotFoundError(f"KB 파일 누락: {KB_EMB_PATH} 또는 {KB_CAT_PATH}")

    kb_embeddings = np.load(KB_EMB_PATH).astype(np.float32)
    kb_embeddings = kb_embeddings / (np.linalg.norm(kb_embeddings, axis=1, keepdims=True) + 1e-12)

    with open(KB_CAT_PATH, "r", encoding="utf-8") as f:
        kb_categories = json.load(f)  # 길이 N

    # KB terms/texts 확보 (가능하면 CSV에서)
    kb_terms, kb_texts = None, None
    if os.path.exists(KB_CSV_PATH):
        df_kb = pd.read_csv(KB_CSV_PATH)
        term_col = "term" if "term" in df_kb.columns else df_kb.columns[0]
        kb_terms = df_kb[term_col].astype(str).tolist()
        if "description" in df_kb.columns:
            kb_texts = (df_kb[term_col].astype(str) + " | " + df_kb["description"].astype(str)).tolist()
        else:
            kb_texts = kb_terms[:]
    else:
        kb_terms = [f"item_{i}" for i in range(len(kb_categories))]
        kb_texts = [str(c) for c in kb_categories]

    # 임베딩 중복 제거 (해시 기반) → 검색 왜곡 방지
    def _dedup_embs(embs: np.ndarray, terms: list, cats: list, texts: list):
        import hashlib
        seen, keep = {}, []
        arr = np.ascontiguousarray(embs)
        for i, row in enumerate(arr):
            h = hashlib.sha256(row.view(np.uint8)).hexdigest()
            if h not in seen:
                seen[h] = True
                keep.append(i)
        return arr[keep], [terms[i] for i in keep], [cats[i] for i in keep], [texts[i] for i in keep]

    kb_embeddings, kb_terms, kb_categories, kb_texts = _dedup_embs(
        kb_embeddings, kb_terms, kb_categories, kb_texts
    )

    print(f"✅ KB 로드 완료 (항목: {len(kb_categories)}개, terms:{len(kb_terms)}개)")

except Exception as e:
    print(f"❌ 치명적 오류: 글로벌 설정 실패: {e}")
    raise

# =====================
# 3. 상태 및 노드 타입
# =====================
class AllergyGraphState(TypedDict):
    image_path: str
    raw_ocr_text: str
    ingredients_to_check: List[str]
    current_ingredient: str
    rag_result: dict
    final_allergens: Set[str]
    final_output_json: str

# =====================
# 4. 노드 구현
# =====================
# --- Node 1: OCR ---

def call_gcp_vision_api(state: AllergyGraphState) -> AllergyGraphState:
    print("\n--- (Node 1: call_gcp_vision_api) ---")
    img_path = state.get("image_path", "")
    print(f"GCP Vision OCR 호출... (이미지: {img_path})")
    if not img_path or not os.path.exists(img_path):
        print("⚠️ 이미지 경로가 없거나 존재하지 않습니다.")
        return {**state, "raw_ocr_text": ""}
    try:
        with io.open(img_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        if response.error.message:
            raise RuntimeError(f"GCP API Error: {response.error.message}")
        raw_text = response.full_text_annotation.text
        print(f"✅ OCR 성공. 텍스트 길이: {len(raw_text)}")
        return {**state, "raw_ocr_text": raw_text}
    except Exception as e:
        print(f"❌ GCP Vision 실패: {e}")
        return {**state, "raw_ocr_text": ""}


# --- API 파서 A: Gemini Structured Output ---

def parse_with_gemini_structured(state: AllergyGraphState) -> AllergyGraphState:
    raw_text = state.get("raw_ocr_text", "")
    if not raw_text.strip():
        return {**state, "ingredients_to_check": [], "final_allergens": set()}

    if not _HAS_GEMINI:
        print("⚠️ google-generativeai 미설치. 빈 결과 반환")
        return {**state, "ingredients_to_check": [], "final_allergens": set()}

    api_key = _get_gemini_api_key()
    if not api_key:
        print("⚠️ Gemini API 키가 없습니다. 환경변수 설정 필요. 빈 결과 반환")
        return {**state, "ingredients_to_check": [], "final_allergens": set()}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    schema = {
        "type": "object",
        "properties": {
            "ingredients_block": {"type": "string"},
            "ingredients_list":  {"type": "array", "items": {"type": "string"}},
            "contains_list":     {"type": "array", "items": {"type": "string"}},
            "cross_contamination_lines": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["ingredients_block", "ingredients_list", "contains_list", "cross_contamination_lines"]
    }

    prompt = f"""
[역할] 너는 한국 식품표시 전문 감리원.
[목표] 아래 OCR 원문에서만 추출하여 JSON으로 반환.

[지시]
- '원재료명' 블록을 한 덩어리 문자열로 그대로 ingredients_block에 넣어라.
- 쉼표/구두점 기준으로 재료를 토큰화한 목록을 ingredients_list에 넣어라.
- '알레르기 유발물질', '...함유', '...포함' 등 표시 라인에 등장하는 항목들을 contains_list에 넣어라.
- '같은 제조시설/교차오염/혼입 가능' 등 문장을 cross_contamination_lines에 원문 그대로 넣어라.
- 원문에 없으면 빈 값/빈 배열을 넣어라. 추측 금지.

[OCR 원문]
```text
{raw_text}
```
"""
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0,
            },
        )
        data = json.loads(resp.text)
    except Exception as e:
        print(f"❌ Gemini 파서 오류: {e}")
        return {**state, "ingredients_to_check": [], "final_allergens": set()}

    def _clean(x: str) -> str:
        x = re.sub(r"\s+", "", x)
        x = x.split("(")[0]
        return normalize_to_std(x)

    ing_list   = [_clean(s) for s in data.get("ingredients_list", []) if s]
    contain_ls = [_clean(s) for s in data.get("contains_list", []) if s]

    filtered_ing = [i for i in ing_list if i and not any(i.startswith(k) for k in IGNORE_KEYWORDS)]
    filtered_con = [c for c in contain_ls if c and not any(c.startswith(k) for k in IGNORE_KEYWORDS)]

    found = set([s for s in filtered_con if s in ALLERGENS_STD_SET])
    queue = sorted(set([*filtered_ing, *filtered_con]))

    print(f"✅ Gemini 파싱 완료: queue={len(queue)} / pre_found={sorted(found)}")
    return {**state, "ingredients_to_check": queue, "final_allergens": found}


# --- API 파서 B: Document AI Custom Extractor ---

def parse_with_docai(state: AllergyGraphState,
                     project_id: str,
                     location: str,
                     processor_id: str) -> AllergyGraphState:
    if not _HAS_DOCAI:
        print("⚠️ google-cloud-documentai 미설치. 빈 결과 반환")
        return {**state, "ingredients_to_check": [], "final_allergens": set()}

    img_path = state.get("image_path", "")
    try:
        client = documentai.DocumentProcessorServiceClient()
        name = client.processor_path(project=project_id, location=location, processor=processor_id)
        with open(img_path, "rb") as f:
            raw_doc = documentai.RawDocument(content=f.read(), mime_type="image/jpeg")
        req = documentai.ProcessRequest(name=name, raw_document=raw_doc)
        result = client.process_document(request=req)
        doc = result.document
    except Exception as e:
        print(f"❌ Document AI 호출 실패: {e}")
        return {**state, "ingredients_to_check": [], "final_allergens": set()}

    ingredients_block = ""
    ingredients_list, contains_list, cross_lines = [], [], []

    for ent in doc.entities:
        t = ent.type_
        val = (ent.mention_text or "").strip()
        if   t == "ingredients_block":        ingredients_block = val
        elif t == "ingredients_item":         ingredients_list.append(val)
        elif t == "allergens_contains_item":  contains_list.append(val)
        elif t == "cross_contamination_line": cross_lines.append(val)

    def _clean(x: str) -> str:
        x = re.sub(r"\s+", "", x)
        x = x.split("(")[0]
        return normalize_to_std(x)

    ing_list   = [_clean(s) for s in ingredients_list if s]
    contain_ls = [_clean(s) for s in contains_list if s]

    filtered_ing = [i for i in ing_list if i and not any(i.startswith(k) for k in IGNORE_KEYWORDS)]
    filtered_con = [c for c in contain_ls if c and not any(c.startswith(k) for k in IGNORE_KEYWORDS)]

    found = set([s for s in filtered_con if s in ALLERGENS_STD_SET])
    queue = sorted(set([*filtered_ing, *filtered_con]))

    print(f"✅ Document AI 파싱 완료: queue={len(queue)} / pre_found={sorted(found)}")
    return {**state, "ingredients_to_check": queue, "final_allergens": found}


# --- Node 2: API 파서 라우터 ---

def parse_text_via_api(state: AllergyGraphState) -> AllergyGraphState:
    print("\n--- (Node 2: parse_text_via_api) [API Parser] ---")
    if USE_API_PARSER == "docai":
        project_id = os.environ.get("DOCAI_PROJECT", "YOUR_GCP_PROJECT")
        location   = os.environ.get("DOCAI_LOCATION", "asia-northeast1")
        processor  = os.environ.get("DOCAI_PROCESSOR_ID", "your-processor-id")
        return parse_with_docai(state, project_id, location, processor)
    else:
        return parse_with_gemini_structured(state)


# --- Node 3: 루프 컨트롤러 ---

def prepare_next_ingredient(state: AllergyGraphState) -> AllergyGraphState:
    print("\n--- (Node 3: prepare_next_ingredient) ---")
    queue = list(state.get("ingredients_to_check", []))
    if not queue:
        print("ℹ️ 남은 항목 없음")
        return state
    nxt = queue.pop(0)
    print(f"다음 검사 대상: '{nxt}' (남은 {len(queue)}개)")
    return {**state, "current_ingredient": nxt, "ingredients_to_check": queue}


# --- RAG 안전 검색 (top-k + 가드룰) ---

def rag_search_topk(query_text: str, k: int = 5, thresh: float = 0.65):
    # 0) 동의어→표준: 질의 자체가 표준 알레르겐이면 바로 확정
    std = normalize_to_std(query_text)
    if std in ALLERGENS_STD_SET:
        return [{"term": std, "category": std, "text": std, "sim": 1.0, "found_by": "alias"}]

    # 1) 쿼리 임베딩은 항상 새로 계산 (부분 일치 캐시 금지)
    q = embedding_model.encode([query_text], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)[0]

    # 2) 코사인 유사도 (정규화 가정)
    sims = kb_embeddings @ q  # (N,)

    # 3) top-k
    k = max(1, min(k, len(sims)))
    top_idx = np.argpartition(-sims, kth=k-1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    results = []
    for i in top_idx:
        results.append({
            "term": kb_terms[i],
            "category": kb_categories[i],
            "text": kb_texts[i],
            "sim": float(sims[i]),
            "found_by": "rag"
        })

    if not results:
        return [{"term": None, "category": "없음", "text": "", "sim": 0.0, "found_by": "none"}]

    # 4) 극단값 보정: 0.99 이상인데도 용어가 다르면 살짝 강등
    r0 = results[0]
    if r0["sim"] >= 0.99:
        if normalize_to_std(r0["term"]) != std and r0["term"] != query_text:
            r0["sim"] = r0["sim"] - 0.05
            results = sorted(results, key=lambda x: -x["sim"])
            r0 = results[0]

    # 5) **용어 일치성 가드**: 핵심어가 다르면 '없음'으로 차단
    if not lexical_consistent(query_text, r0["term"]):
        return [{"term": None, "category": "없음", "text": "", "sim": float(r0["sim"]), "found_by": "lex_guard"}]

    # 6) 임계치 미달이면 '없음'
    if r0["sim"] < thresh:
        return [{"term": None, "category": "없음", "text": "", "sim": float(r0["sim"]), "found_by": "below_thresh"}]

    return results[:k]


# --- Node 4: RAG 검색 ---

def rag_search(state: AllergyGraphState) -> AllergyGraphState:
    print("--- (Node 4: rag_search) ---")
    ingredient = state.get("current_ingredient", "")

    cand_list = rag_search_topk(ingredient, k=5, thresh=0.65)
    top = cand_list[0]

    found = top["category"]
    conf  = float(top["sim"])
    by    = top.get("found_by")
    print(f"RAG 검색: '{ingredient}' → '{found}' (유사도 {conf:.4f}, by={by})")

    return {**state, "rag_result": {"confidence": conf, "found_allergen": found}}


# --- Node 5: LLM Fallback (Zero-Shot) ---

def llm_fallback(state: AllergyGraphState) -> AllergyGraphState:
    print("--- (Node 5: llm_fallback) [NLI Zero-Shot] ---")
    ingredient = state.get("current_ingredient", "")
    try:
        resp = nli_pipeline(ingredient, list(ALLERGENS_STD_SET) + ["관련 없음"])
        top_label, top_score = resp['labels'][0], float(resp['scores'][0])
        print(f"NLI 응답: Label='{top_label}', Score={top_score:.4f}")
        if top_label in ALLERGENS_STD_SET and top_score >= NLI_FALLBACK_THRESHOLD:
            return {**state, "rag_result": {"confidence": top_score, "found_allergen": top_label}}
        return {**state, "rag_result": {"confidence": 1.0, "found_allergen": "없음"}}
    except Exception as e:
        print(f"❌ NLI Fallback 오류: {e}")
        return {**state, "rag_result": {"confidence": 1.0, "found_allergen": "오류"}}


# --- Node 6: 결과 취합 ---

def update_final_list(state: AllergyGraphState) -> AllergyGraphState:
    print("--- (Node 6: update_final_list) ---")
    result_allergen = state.get("rag_result", {}).get("found_allergen", "")
    if result_allergen in ALLERGENS_STD_SET:
        s = set(state.get("final_allergens", set()))
        s.add(result_allergen)
        print(f"✅ 유효 알레르기 추가: '{result_allergen}' → {sorted(s)}")
        return {**state, "final_allergens": s}
    print(f"ℹ️ 표준 알레르기 아님 또는 '없음': '{result_allergen}' (무시)")
    return state


# --- Node 7: 종료 ---

def finalize_processing(state: AllergyGraphState) -> AllergyGraphState:
    print("\n--- (Node 7: finalize_processing) ---")
    final_set = set(state.get("final_allergens", set()))
    final_list = sorted(list(final_set))
    final_json = json.dumps(final_list, ensure_ascii=False)
    print(f"🎉 최종 결과: {final_json}")
    return {**state, "final_output_json": final_json}


# =====================
# 5. 엣지(Edge) 라우터
# =====================

def route_after_parse(state: AllergyGraphState) -> str:
    if state.get("ingredients_to_check"):
        return "has_ingredients"
    return "no_ingredients"


def route_rag_result(state: AllergyGraphState) -> str:
    conf = state.get("rag_result", {}).get("confidence", 0.0)
    allergen = state.get("rag_result", {}).get("found_allergen", "")

    # '없음'이면 폴백 불필요 → 바로 다음 단계로(추가 안 되고 넘어감)
    if allergen == "없음":
        print("  -> [RAG 결과 없음] update_final_list (폴백 생략)")
        return "rag_success"

    if conf >= RAG_CONFIDENCE_THRESHOLD and allergen in ALLERGENS_STD_SET:
        print("  -> [RAG 성공] update_final_list")
        return "rag_success"

    print("  -> [RAG 불확실] llm_fallback")
    return "needs_llm_fallback"


def check_remaining_ingredients(state: AllergyGraphState) -> str:
    if state.get("ingredients_to_check"):
        print("  -> [항목 남음] prepare_next_ingredient")
        return "has_more_ingredients"
    print("  -> [항목 없음] finalize_processing")
    return "all_ingredients_done"


# =====================
# 6. 그래프 빌드
# =====================
print("\n--- LangGraph 워크플로우 빌드 시작 ---")
workflow = StateGraph(AllergyGraphState)

# 노드 등록
workflow.add_node("call_gcp_vision_api", call_gcp_vision_api)
workflow.add_node("parse_text_via_api", parse_text_via_api)
workflow.add_node("prepare_next_ingredient", prepare_next_ingredient)
workflow.add_node("rag_search", rag_search)
workflow.add_node("llm_fallback", llm_fallback)
workflow.add_node("update_final_list", update_final_list)
workflow.add_node("finalize_processing", finalize_processing)

# 엣지 연결
workflow.set_entry_point("call_gcp_vision_api")
workflow.add_edge("call_gcp_vision_api", "parse_text_via_api")

# parse → 조건부 분기
workflow.add_conditional_edges(
    "parse_text_via_api",
    route_after_parse,
    {"has_ingredients": "prepare_next_ingredient", "no_ingredients": "finalize_processing"}
)

# 루프 본체
workflow.add_edge("prepare_next_ingredient", "rag_search")
workflow.add_conditional_edges(
    "rag_search",
    route_rag_result,
    {"rag_success": "update_final_list", "needs_llm_fallback": "llm_fallback"}
)
workflow.add_edge("llm_fallback", "update_final_list")
workflow.add_conditional_edges(
    "update_final_list",
    check_remaining_ingredients,
    {"has_more_ingredients": "prepare_next_ingredient", "all_ingredients_done": "finalize_processing"}
)
workflow.add_edge("finalize_processing", END)

# 컴파일
app = workflow.compile()
print("--- ✅ LangGraph 워크플로우 컴파일 완료 ---")


# =====================
# 7. 디버그/검증 유틸 (선택)
# =====================

def kb_self_check(max_show: int = 5):
    """중복 임베딩 그룹/샘플 표시"""
    import hashlib
    groups = {}
    arr = np.ascontiguousarray(kb_embeddings)
    for i, row in enumerate(arr):
        h = hashlib.sha256(row.view(np.uint8)).hexdigest()
        groups.setdefault(h, []).append(i)
    dup_groups = {h:idxs for h,idxs in groups.items() if len(idxs) > 1}
    print(f"[SELF-CHECK] 중복 임베딩 그룹 수: {len(dup_groups)}")
    for h, idxs in list(dup_groups.items())[:max_show]:
        names = [kb_terms[i] for i in idxs]
        cats  = [kb_categories[i] for i in idxs]
        print(f"  - size={len(idxs)} | terms={names[:5]} | cats={cats[:5]}")


# =====================
# 8. 테스트 실행 (예시)
# =====================
if __name__ == "__main__":
    print("\n--- [Test Run: GCP OCR + API Parser + RAG + NLI] ---")

    # (선택) KB 중복 체크
    try:
        kb_self_check()
    except Exception as e:
        print(f"[SELF-CHECK] 실패: {e}")

    # 예시 이미지 경로
    test_image = os.environ.get("ALLER_GUARD_TEST_IMAGE", r"C:\\Users\\MYNOTE\\AllerGuard\\Data\\김광무_118.jpg")
    if not os.path.exists(test_image):
        print(f"⚠️ 테스트 이미지가 존재하지 않습니다: {test_image}")
    test_input = {"image_path": test_image}

    try:
        final_state = app.invoke(test_input, {"recursion_limit": 1000})
        print("\n최종 반환 JSON:")
        print(final_state.get('final_output_json', ''))
    except Exception as e:
        print(f"❌ 실행 오류: {e}")

