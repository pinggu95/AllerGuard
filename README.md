# AllerGuard

## ☝🏻 프로젝트 개요
- **과정/회차:** 생성 AI 응용 서비스 개발자 양성 과정 (4회차)
- **프로젝트 기간:** 2025.09.03 ~ 2025.09.16
- **일자(기획서 기준):** 2025.08.29
- **팀장:** 차지예
- **구성원(가나다순):** 김광무, 정주환, 정형웅, 차지예
- **프로젝트명:** **AllerGuard**
- **주제:** VLM을 이용한 **식품 라벨 사진에서 알레르기 자동 판별 및 경고** 개발

---

## 프로젝트 소개
AllerGuard는 사용자가 촬영/업로드한 **식품 라벨 이미지**를 분석하여 **알레르기 유발 성분을 자동 판별**하고 즉각 **주의/경고 신호**를 제공하는 서비스입니다. 작은 글씨, 다국어 표기, **대체명/파생명**(예: 우유→유청/카제인, 땅콩→groundnut/arachis)과 **부정·주의 표현**(non-dairy, may contain, same facility), **교차오염 문구** 등으로 확인이 어려운 현실을 해결합니다.

---

## 프로젝트 목표
- **안전한 식품 선택**을 위해 사진 속 **특정 알레르기 유발 물질 자동 판별** 및 **즉각 경고 제공**
- 결과를 **노랑(개인화)/빨강(위험)**의 직관 체계와 **명확한 텍스트**로 제공해 **의사결정 시간 단축**
- **기본 22종** 외 사용자 프로필 기반의 **개인 맞춤형 경고** 제공
- VLM 파인튜닝으로 **Precision/Recall/F1** 성능 개선 및 **모델/데모 공개**(Hugging Face, Gradio)

---

## 지원 기능
- **라벨 이미지 입력:** 촬영/업로드(단일·여러 장) 지원
- **OCR + 멀티모달 판별:** 이미지+텍스트 동시 분석으로 알레르겐, **대체명/부정·주의 표현/교차오염 문구**까지 감지
- **경고 UI:**
  - **빨강:** 고위험 알레르겐 명시 탐지
  - **노랑(개인화):** 사용자 등록 알레르기와 매칭 시 주의
  - 근거 텍스트/영역 하이라이트 및 **간단 설명** 제공
- **개인화 프로필:** 사용자 알레르기 항목 등록·관리
- **이력 조회:** 최근 스캔 결과 확인(옵션)
- **모델 정보 노출:** 사용 모델/버전·평가 지표 공개(옵션)
- **배포:** Gradio Demo 및 모델 HF 업로드/연동

---

## 사용 스택
- **AI/모델:** Vision-Language Model(후보 비교·파인튜닝), PyTorch, Hugging Face Transformers
- **OCR:** Tesseract 또는 PaddleOCR (환경에 맞게 선택)
- **데이터/학습:** 정제 데이터셋, 학습/검증 파이프라인, 지표(Precision·Recall·F1)
- **서비스/UI:** Gradio(프로토타입), FastAPI/Streamlit(선택)
- **배포/협업:** Hugging Face Hub, Git/GitHub
- **기타:** 이미지 전처리(OpenCV/Pillow), 텍스트 정규화
- **최종모델**
  - OCR : GCP OCR API
  - NLI : klue/roberta-base
  - LLM : OpenAI GPT-4.1
  - Search : Tavily
  - UI : Gradio
  - Framework : LangGraph, LangChain, sentence_transformers, google.cloud, Pandas, Numpy 

---

## 소스설명
- **AllerGuard_gradio.py:** Gradio UI
- **Allerguard_V1.py:** 메인 소스 코드
- **domestic_allergy_rag_knowledge_1000.csv:** 선수집된 알러지 성분 사전
- **20250916_테스트_및_검증용_식품_라벨링.xlsx**

