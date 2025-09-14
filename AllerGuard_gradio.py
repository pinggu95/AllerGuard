# -*- coding: utf-8 -*-
# app_gradio.py

import io, re, json, html, tempfile, ast, traceback
from contextlib import redirect_stdout
from PIL import Image, ImageOps
import gradio as gr
import Allerguard_V1

# ===============================
# 설정: 표준 알레르겐 (UI 분류에 사용)
# ===============================
ALLERGENS_STD_SET = set([
    "알류","우유","메밀","땅콩","대두","밀","잣","호두",
    "게","새우","오징어","고등어","조개류","복숭아","토마토",
    "닭고기","돼지고기","쇠고기","아황산류"
])

# ===============================
# High-Contrast UI + 위험/주의/안전
# ===============================
CUSTOM_CSS = """
:root{
  --bg:#f4f6fb;      --panel:#ffffff;  --text:#111827;  --muted:#4b5563;
  --border:#e5e7eb;  --brand:#10a37f;  --brand-700:#0e8e6f;
  --danger:#dc2626;  --safe:#059669;   --warn:#f97316;
  --code-bg:#0f172a; --code-fg:#e5e7eb;
}
html, body, .gradio-container { background: var(--bg) !important; color: var(--text); }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.headerbar{
  background:#111827; color:#fff; border-radius:14px; padding:18px 20px; margin:18px 0 8px;
  border:1px solid #0b1220; box-shadow:0 8px 28px rgba(2,6,23,.25);
  display:flex; align-items:center; justify-content:space-between; gap:12px;
}
.headerbar .title{ font-weight:900; font-size:26px; letter-spacing:-.2px; color:#fff; }
.headerbar .subtitle{ color:#e5e7eb; font-size:14px; }
.card{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px;
       box-shadow:0 10px 28px rgba(2,6,23,.06); }
.section-title{ font-weight:900; margin-bottom:8px; display:flex; align-items:center; gap:10px; }
.section-title .dot{ width:8px; height:8px; border-radius:50%; background:var(--brand); display:inline-block; }
.run_btn > button{ width:100%; background:var(--brand) !important; color:#fff !important;
  font-weight:800 !important; border:none !important; letter-spacing:.2px; }
.run_btn > button:hover{ background:var(--brand-700) !important; }
.summary{
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:14px 16px; border-radius:12px; border:1px dashed var(--border); background:#f8fafc;
}
.summary .left{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.badge{ display:inline-block; padding:6px 10px; border-radius:999px; font-size:12.5px; font-weight:900;
  background:#e6f5ef; color:var(--brand); border:1px solid #cbeedf; }
.badge.danger{ background:#fee2e2; color:#dc2626; border-color:#fecaca; }
.badge.safe  { background:#dcfce7; color:#059669; border-color:#bbf7d0; }
.badge.warn  { background:#ffedd5; color:#f97316; border-color:#fdba74; }
.state-text-danger{ color:#dc2626; font-weight:900; }
.state-text-warn  { color:#f97316; font-weight:900; }
.state-text-safe  { color:#059669; font-weight:900; }
.state-text-ok    { color:#10a37f; font-weight:900; }
.pills{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
.pill{ display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px;
  background:#fee2e2; color:#b91c1c; border:1px solid #fecaca; font-weight:700; font-size:13px; }
.pill.warn{ background:#ffedd5; color:#c2410c; border-color:#fdba74; }
.pill.safe{ background:#dcfce7; color:#0f7a56; border-color:#bbf7d0; }
.codebox{ background:var(--code-bg); color:var(--code-fg); border:1px solid #0b1220; border-radius:10px;
  padding:12px; white-space:pre-wrap; font-size:13px; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  max-height:350px; overflow:auto; }
.kv{ font-family: ui-monospace, Menlo, Consolas, "Liberation Mono", monospace; }
.search-row{ display:flex; gap:10px; align-items:center; }
"""

# --------- 로그 파서 ---------
RAG_LINE      = re.compile(r"RAG 검색: '([^']+)' \(유사도:\s*([0-9.]+)\) -> 매핑: '([^']+)'")
QUEUE_LINE    = re.compile(r"최종 RAG 검사 큐.*?:\s*(\[[^\]]*\])")
NLI_Q_LINE    = re.compile(r"NLI Fallback:\s*'([^']+)'\s*분류 요청")
NLI_RES_LINE  = re.compile(r"NLI 응답:\s*Label='([^']+)',\s*Score=([0-9.]+)")

def parse_logs(raw_logs: str):
    ingredients, rag_hits, nli_hits = [], [], []
    try:
        m = QUEUE_LINE.search(raw_logs or "")
        if m:
            ingredients = ast.literal_eval(m.group(1))
    except Exception:
        ingredients = []
    try:
        for ing, sim, al in RAG_LINE.findall(raw_logs or ""):
            rag_hits.append((ing, float(sim), al))
    except Exception:
        rag_hits = []
    try:
        current_ing = None
        for line in (raw_logs or "").splitlines():
            mq = NLI_Q_LINE.search(line)
            if mq:
                current_ing = mq.group(1); continue
            mr = NLI_RES_LINE.search(line)
            if mr and current_ing:
                label, score = mr.group(1), float(mr.group(2))
                nli_hits.append((current_ing, score, label))
                current_ing = None
    except Exception:
        nli_hits = []
    return ingredients, rag_hits, nli_hits

def safe_load_allergen_list(final_json_str_or_obj):
    if isinstance(final_json_str_or_obj, list):
        return final_json_str_or_obj
    if isinstance(final_json_str_or_obj, str):
        s = final_json_str_or_obj.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        return []
    return []

# ===============================
# A안: "원재료 -> 알레르겐" 패턴 파싱(권장)
# ===============================
def build_categories(final_allergens, ingredients, rag_hits, nli_hits,
                     rag_warn_low=0.65, rag_warn_high=0.85,
                     nli_warn_low=0.30, nli_warn_high=0.50):

    danger_allergens = []
    mapped_ingredients = set()

    # 1) 최종 JSON 정규화: "원재료 -> 알레르겐" 문자열도 해석
    for a in (final_allergens or []):
        if isinstance(a, str) and "->" in a:
            src, dst = [s.strip() for s in a.split("->", 1)]
            if dst in ALLERGENS_STD_SET:
                danger_allergens.append(dst)
                mapped_ingredients.add(src)  # 안전 목록에서 제외
        elif a in ALLERGENS_STD_SET:
            danger_allergens.append(a)

    # 중복 제거(순서 보존)
    danger_allergens = list(dict.fromkeys(danger_allergens))

    # 2) 주의(경고) 후보
    warn_from_rag = []
    for ing, sim, al in (rag_hits or []):
        if al in ALLERGENS_STD_SET and rag_warn_low <= sim < rag_warn_high:
            warn_from_rag.append(f"{ing} → {al} (유사도 {sim:.2f})")

    warn_from_nli = []
    for ing, score, label in (nli_hits or []):
        if label in ALLERGENS_STD_SET and nli_warn_low <= score < nli_warn_high:
            warn_from_nli.append(f"{ing} → {label} (NLI {score:.2f})")

    warn_items = warn_from_rag + warn_from_nli

    # 3) 안전 항목 계산 (이미 위험/주의에 쓰인 원재료 제외)
    used_ingredients = set(mapped_ingredients)
    used_ingredients.update([i.split(" → ")[0].strip() for i in warn_items if "→" in i])
    safe_items = [i for i in (ingredients or []) if i not in used_ingredients]

    return danger_allergens, warn_items, safe_items

def _build_pills(items, cls=""):
    if not items:
        tone = "safe" if cls == "safe" else ("warn" if cls == "warn" else "danger")
        empty_text = {"danger":"표시할 항목이 없습니다.",
                      "warn":"주의 대상이 없습니다.",
                      "safe":"표시할 항목이 없습니다."}[tone]
        return f"<div class='kv' style='color:#6b7280'>{empty_text}</div>"
    pill_cls = f"pill {cls}".strip()
    return "<div class='pills'>" + "".join([f"<span class='{pill_cls}'>{html.escape(str(x))}</span>" for x in items]) + "</div>"

# ===== (신규) 더 잘 읽힌 쪽 선택 스코어러 =====
def _score_run(final_allergens, ingredients):
    # 성분 토큰 수 + (확정 알레르겐 수 * 2) 가중치
    return len(ingredients) * 1.0 + len([a for a in final_allergens if (isinstance(a,str) and a in ALLERGENS_STD_SET)]) * 2.0

# ===============================
# 메인 핸들러 (자동/수동 미러 보정 포함)
# ===============================
def analyze_image(img: Image.Image, do_mirror: bool, auto_mirror: bool, using_llm_api_chk: bool, parser_type:str):
    try:
        if img is None:
            status = ("<div class='summary'>"
                      "<div class='left'><span class='badge'>상태</span><span class='state-text-danger'>이미지가 필요합니다</span></div>"
                      "<div class='kv'>N/A</div></div>")
            return status, "", "", "", {"error":"이미지를 업로드해 주세요."}, "이미지를 업로드해 주세요.", []

        # ✅ EXIF 회전 보정
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        if auto_mirror:
            # 1) 원본 실행
            sio1 = io.StringIO()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp1:
                img.convert("RGB").save(tmp1.name, "JPEG")
                path1 = tmp1.name
            with redirect_stdout(sio1):
                state1 = Allerguard_V1.app.invoke({"image_path": path1,"using_llm_api_chk":using_llm_api_chk,"text_parser":parser_type,}, {"recursion_limit": 2000})
            logs1 = sio1.getvalue().strip()
            fj1 = state1.get("final_output_json", "[]")
            fa1 = safe_load_allergen_list(fj1)
            ing1, rag1, nli1 = parse_logs(logs1)
            score1 = _score_run(fa1, ing1)

            # 2) 좌우반전 실행
            img_m = ImageOps.mirror(img)
            sio2 = io.StringIO()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:
                img_m.convert("RGB").save(tmp2.name, "JPEG")
                path2 = tmp2.name
            with redirect_stdout(sio2):
                state2 = Allerguard_V1.app.invoke({"image_path": path2,"using_llm_api_chk":using_llm_api_chk,"text_parser":parser_type,}, {"recursion_limit": 2000})
            logs2 = sio2.getvalue().strip()
            fj2 = state2.get("final_output_json", "[]")
            fa2 = safe_load_allergen_list(fj2)
            ing2, rag2, nli2 = parse_logs(logs2)
            score2 = _score_run(fa2, ing2)

            if score2 > score1:
                final_allergens = fa2; ingredients, rag_hits, nli_hits = ing2, rag2, nli2
                raw_logs = "[자동미러] 좌우반전 결과 채택\n" + logs2
            else:
                final_allergens = fa1; ingredients, rag_hits, nli_hits = ing1, rag1, nli1
                raw_logs = "[자동미러] 원본 결과 채택\n" + logs1

        else:
            # 수동 미러 보정
            if do_mirror:
                img = ImageOps.mirror(img)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                img.convert("RGB").save(tmp.name, "JPEG")
                tmp_path = tmp.name

            sio = io.StringIO()
            with redirect_stdout(sio):
                state = Allerguard_V1.app.invoke({"image_path": tmp_path,"using_llm_api_chk":using_llm_api_chk,"text_parser":parser_type,}, {"recursion_limit": 2000})
            raw_logs = sio.getvalue().strip()

            final_json = state.get("final_output_json", "[]")
            final_allergens = safe_load_allergen_list(final_json)
            ingredients, rag_hits, nli_hits = parse_logs(raw_logs)

        # 결과 요약/표시
        danger_list, warn_items, safe_items = build_categories(final_allergens, ingredients, rag_hits, nli_hits)
        status = (
            "<div class='summary'>"
            "<div class='left'>"
            "<span class='badge danger'>위험</span>"
            f"<span class='state-text-danger'>감지됨 · {len(danger_list)}종</span>"
            "<span class='badge warn'>주의</span>"
            f"<span class='state-text-warn'>{len(warn_items)}건</span>"
            "<span class='badge safe'>안전</span>"
            f"<span class='state-text-safe'>{len(safe_items)}건</span>"
            "</div>"
            "<div class='kv state-text-ok'>결과코드: OK</div>"
            "</div>"
        )

        danger_html = _build_pills(danger_list, cls="")
        warn_html   = _build_pills(warn_items,   cls="warn")
        safe_html   = _build_pills(safe_items[:20], cls="safe")

        json_view_obj = final_allergens
        log_view_html = f"<div class='codebox'>{html.escape(raw_logs or '로그가 없습니다.')}</div>"

        return status, danger_html, warn_html, safe_html, json_view_obj, log_view_html, warn_items

    except Exception as e:
        err = f"[ERROR] {str(e)}\n\n" + traceback.format_exc()
        safe_status = ("<div class='summary'>"
                       "<div class='left'><span class='badge danger'>에러</span>"
                       "<span class='state-text-danger'>처리 중 오류가 발생했습니다</span></div>"
                       "<div class='kv'>ERROR</div></div>")
        return safe_status, "", "", "", {"error":"내부 오류 발생"}, f"<div class='codebox'>{html.escape(err)}</div>", []

def filter_caution(query, warn_items):
    try:
        if not warn_items:
            return "<div class='kv' style='color:#6b7280'>주의 대상이 없습니다.</div>"
        q = (query or "").strip()
        if not q:
            return _build_pills(warn_items, cls="warn")
        filtered = [x for x in warn_items if q.lower() in str(x).lower()]
        if not filtered:
            return "<div class='kv' style='color:#6b7280'>검색 결과가 없습니다.</div>"
        return _build_pills(filtered, cls="warn")
    except Exception as e:
        return f"<div class='kv' style='color:#b91c1c'>검색 오류: {html.escape(str(e))}</div>"

# ===============================
# UI
# ===============================
with gr.Blocks(title="식품 알레르기 감지 · High Contrast", css=CUSTOM_CSS, theme=gr.themes.Citrus()) as demo:
    gr.HTML("""
      <div class="headerbar">
        <div class="title">식품 알레르기 감지</div>
        <div class="subtitle">이미지 업로드 후 <b>분석 실행</b>을 누르면 결과가 표시됩니다.</div>
      </div>
    """)

    with gr.Row():
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("<div class='section-title'><span class='dot'></span>입력</div>")
                inp = gr.Image(type="pil", label="성분표 이미지 업로드", height=360)
                do_mirror_chk = gr.Checkbox(label="좌우반전(미러) 보정", value=False)
                auto_mirror_chk = gr.Checkbox(label="자동 미러 감지(원본/반전 비교, 2회 실행)", value=False)
                using_llm_api_chk = gr.Checkbox(label="원재료 19종 분류(LLM API)", value=False)
                with gr.Row():
                    run_btn_by_regex = gr.Button("분석 실행(REGEX)", elem_id="run_btn_by_regex", elem_classes=["run_btn"])
                    run_btn_by_llmapi = gr.Button("분석 실행(LLM API)", elem_id="run_btn_by_llmapi", elem_classes=["run_btn"])
                    parser_type_by_regex = gr.Textbox(value="text_parser_by_regex", visible=False)
                    parser_type_by_llm = gr.Textbox(value="text_parser_by_llm", visible=False)
        with gr.Column(scale=8, min_width=520):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("<div class='section-title'><span class='dot'></span>요약</div>")
                status_html = gr.HTML()

    with gr.Row():
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 위험 (확정 알레르겐)")
                danger_html = gr.HTML()
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 주의 (추가 확인 권장)")
                with gr.Row(elem_classes=["search-row"]):
                    warn_search = gr.Textbox(label="주의 항목 검색", placeholder="예: 젤라틴, 향료, 유사도 ...", scale=2)
                warn_html = gr.HTML()
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### 안전 (문제 성분 없음)")
                safe_html = gr.HTML()

    with gr.Row():
        with gr.Column(scale=12):
            with gr.Group(elem_classes=["card"]):
                with gr.Tabs():
                    with gr.Tab("원본 JSON"):
                        json_view = gr.JSON()
                    with gr.Tab("실행 로그"):
                        log_view = gr.HTML()

    warn_state = gr.State([])

    run_btn_by_regex.click(
        fn=analyze_image,
        inputs=[inp, do_mirror_chk, auto_mirror_chk, using_llm_api_chk, parser_type_by_regex],
        outputs=[status_html, danger_html, warn_html, safe_html, json_view, log_view, warn_state],
        api_name="analyze"
    )
    
    run_btn_by_llmapi.click(
        fn=analyze_image,
        inputs=[inp, do_mirror_chk, auto_mirror_chk, using_llm_api_chk, parser_type_by_llm],
        outputs=[status_html, danger_html, warn_html, safe_html, json_view, log_view, warn_state],
        api_name="analyze"
    )

    warn_search.input(
        fn=filter_caution,
        inputs=[warn_search, warn_state],
        outputs=[warn_html]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
