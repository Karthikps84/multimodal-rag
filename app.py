"""
==========================================================
Streamlit UI — Warm & Friendly theme, two tabs
==========================================================
Run:  streamlit run app.py
==========================================================
"""

import time
from pathlib import Path

import streamlit as st

from rag_engine import RAGEngine
import config

st.set_page_config(page_title="Docs Assistant", page_icon="🌻", layout="wide", initial_sidebar_state="collapsed")

SUPPORTED_EXT = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".pptx", ".txt", ".md"}

# ── Warm & friendly design tokens ────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&family=Nunito:wght@400;500;600;700&display=swap');

:root {
  --bg:        #FFF8F0;
  --bg-panel:  #FFF2E2;
  --card:      #FFFFFF;
  --border:    #FBE0C4;

  --coral:     #FF8A65;
  --coral-lt:  #FFE0D2;
  --amber:     #FFB74D;
  --amber-lt:  #FFF0D6;
  --sage:      #8BBF9F;
  --sage-lt:   #E3F1E7;

  --tx-hi:     #4A3728;
  --tx-mid:    #7A6355;
  --tx-lo:     #A78E7E;

  --r:  18px;
  --rs: 12px;
  --shadow: 0 2px 10px rgba(180, 120, 80, 0.08);
  --shadow-hover: 0 4px 16px rgba(180, 120, 80, 0.13);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'Nunito', sans-serif !important;
  color: var(--tx-hi) !important;
}
h1, h2, h3, .hero-title, .brand-name { font-family: 'Quicksand', sans-serif !important; }

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--coral-lt); border-radius: 6px; }

/* Tabs */
[data-testid="stTabs"] button {
  font-family: 'Quicksand', sans-serif !important; font-weight: 700 !important;
  font-size: 15px !important; color: var(--tx-mid) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--coral) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--coral) !important; }

.sec-label {
  font-size: 11px; font-weight: 700; letter-spacing: .6px; text-transform: uppercase;
  color: var(--tx-lo); margin: 18px 0 8px;
}

.icard {
  background: var(--card); border: 1px solid var(--border); border-radius: var(--rs);
  padding: 10px 13px; margin-bottom: 6px; display: flex; align-items: center;
  justify-content: space-between; box-shadow: var(--shadow);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.icard:hover { transform: translateY(-1px); box-shadow: var(--shadow-hover); }
.icard-lbl { font-size: 13px; color: var(--tx-mid); }
.icard-val { font-size: 12px; font-weight: 600; color: var(--tx-hi); text-align: right; }

.pill {
  display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 600;
  padding: 3px 10px; border-radius: 20px; transition: transform 0.12s ease;
}
.pill:hover { transform: scale(1.04); }
.pill-coral { background: var(--coral-lt); color: #C0603F; }
.pill-sage  { background: var(--sage-lt);  color: #4E8362; }
.pill-amber { background: var(--amber-lt); color: #B77B1E; }

[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 2px dashed var(--coral-lt) !important;
  border-radius: var(--r) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--coral) !important; }

[data-testid="stButton"] > button {
  width: 100% !important;
  background: linear-gradient(135deg, var(--coral), var(--amber)) !important;
  color: #fff !important; font-weight: 700 !important; border: none !important;
  border-radius: var(--rs) !important; font-size: 14px !important; padding: 10px 16px !important;
  box-shadow: var(--shadow) !important; transition: 0.15s ease !important;
}
[data-testid="stButton"] > button:hover { box-shadow: var(--shadow-hover) !important; transform: translateY(-1px) !important; }
[data-testid="stButton"] > button:disabled { background: #F0E4D8 !important; color: var(--tx-lo) !important; box-shadow: none !important; }

/* Secondary (outline) buttons -- used for Cancel */
.btn-secondary [data-testid="stButton"] > button {
  background: var(--card) !important; color: var(--tx-mid) !important;
  border: 1.5px solid var(--border) !important; box-shadow: none !important;
}

[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--coral), var(--amber)) !important; border-radius: 6px !important; }
[data-testid="stProgress"] > div { background: var(--coral-lt) !important; border-radius: 6px !important; }

.hero {
  background: var(--card); border: 1px solid var(--border); border-radius: 22px;
  padding: 26px 32px 22px; margin-bottom: 18px; box-shadow: var(--shadow);
  position: relative; overflow: hidden;
}
.hero::after {
  content: ''; position: absolute; top: -60px; right: -40px; width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(255,138,101,0.10) 0%, transparent 70%);
  pointer-events: none;
}
.hero-title { font-size: 26px; font-weight: 700; color: var(--tx-hi); margin: 0 0 4px; }
.hero-sub { font-size: 14px; color: var(--tx-mid); margin: 0; }

[data-testid="stChatMessage"] {
  border-radius: var(--r) !important; margin-bottom: 14px !important;
  animation: fadeIn 0.25s ease-out !important;
  transition: box-shadow 0.15s ease, transform 0.15s ease !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: linear-gradient(135deg, #FFF0E8, #FFF7EE) !important;
  border: 1px solid var(--coral-lt) !important;
  box-shadow: var(--shadow) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow) !important;
}
[data-testid="stChatMessage"]:hover { box-shadow: var(--shadow-hover) !important; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}

.streaming-cursor { color: var(--coral); font-weight: 700; }
[data-testid="stChatInputContainer"] {
  background: var(--card) !important; border: 2px solid var(--coral-lt) !important;
  border-radius: 18px !important;
}
[data-testid="stChatInputContainer"]:focus-within { border-color: var(--coral) !important; }

[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: var(--rs) !important; }
[data-testid="stExpander"] summary { font-size: 12px !important; color: var(--tx-mid) !important; }

.meta-row { display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.meta-dot { font-size: 12px; color: var(--tx-lo); }

hr { border-color: var(--border) !important; margin: 12px 0 !important; }
code { font-family: monospace !important; background: var(--amber-lt) !important; border-radius: 5px !important; padding: 1px 6px !important; color: #B77B1E !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--rs) !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] { background-color: var(--coral) !important; }

.task-card {
  background: var(--card); border: 2px solid var(--coral-lt); border-radius: var(--r);
  padding: 20px 22px; margin-bottom: 16px; box-shadow: var(--shadow);
}
.task-card-title { font-size: 15px; font-weight: 700; color: var(--tx-hi); margin-bottom: 4px; }
.task-card-sub { font-size: 12px; color: var(--tx-mid); margin-bottom: 12px; }

.empty-state {
  background: var(--card); border: 2px dashed var(--coral-lt); border-radius: 22px;
  padding: 48px 32px; text-align: center; margin: 12px 0;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

TASK_LABEL = {
    "local": "📚 knowledge base", "web": "🌐 web search", "analytics": "📊 analytics",
    "summarize": "📝 summary", "extract": "🗂️ extraction", "compare": "⚖️ comparison",
}
TASK_COLOR = {
    "local": "sage", "web": "amber", "analytics": "coral",
    "summarize": "coral", "extract": "coral", "compare": "amber",
}
ICON = {"pdf": "📕", "docx": "📘", "xlsx": "📗", "xls": "📗", "csv": "📊", "pptx": "📙", "txt": "📄", "md": "📝"}


def save_uploaded_files(files, data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    saved, skipped = 0, []
    for f in files:
        if Path(f.name).suffix.lower() not in SUPPORTED_EXT:
            skipped.append(f.name)
            continue
        (data_dir / f.name).write_bytes(f.getbuffer())
        saved += 1
    return saved, skipped


@st.cache_resource(show_spinner=False)
def get_engine():
    """Builds/loads the engine with a real progress bar tied to actual
    pipeline stages (doc loading, chunking, each embed batch)."""
    holder = st.empty()
    bar = holder.progress(0, text="Starting up...")

    def on_progress(pct, msg):
        bar.progress(min(pct, 100) / 100, text=msg)

    rag = RAGEngine()
    rag.initialize(on_progress=on_progress)

    holder.empty()
    return rag


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("messages", []), ("history", []), ("active_task", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


def run_task(question: str, forced_route: str, target_files: list):
    """Shared execution path for both free-chat and the task forms below.
    Streams tokens live for local/web/summarize/compare; falls back to a
    blocking call for extract/analytics (need the full output before it
    can be parsed into a table) and for cache hits (already instant)."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        t0 = time.time()
        prep = engine.prepare_stream(question, target_files=target_files, forced_route=forced_route)

        if prep.get("cached"):
            result = prep["result"]
            answer, route, sources, ana = result["answer"], result.get("route_type", "local"), result.get("sources", []), result.get("analytics_result")
            if ana and ana.get("success") and ana.get("result_type") in ("dataframe", "series"):
                st.markdown(answer)
                st.dataframe(ana["value"], use_container_width=True, hide_index=True)
            else:
                st.markdown(answer)

        elif prep["streamable"]:
            placeholder = st.empty()
            answer = ""
            for chunk in prep["chain"].stream(prep["inputs"]):
                answer += chunk
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
            route, sources, ana = prep["route"], prep["sources"], None
            engine.finalize(question, answer, route, sources, target_files=target_files, forced_route=forced_route)

        else:
            with st.spinner("Working on it..."):
                result = engine.ask(question, target_files=target_files, forced_route=forced_route)
            answer, route, sources, ana = result["answer"], result.get("route_type", forced_route or "local"), result.get("sources", []), result.get("analytics_result")
            if ana and ana.get("success") and ana.get("result_type") in ("dataframe", "series"):
                st.markdown(answer)
                st.dataframe(ana["value"], use_container_width=True, hide_index=True)
            else:
                st.markdown(answer)

        elapsed = time.time() - t0
        st.markdown(
            f'<div class="meta-row">'
            f'<span class="pill pill-{TASK_COLOR.get(route, "coral")}">{TASK_LABEL.get(route, route)}</span>'
            f'<span class="meta-dot">{elapsed:.1f}s</span></div>',
            unsafe_allow_html=True,
        )
        if sources:
            with st.expander(f"📎 {len(sources)} source(s)"):
                for s in sources:
                    st.markdown(f"- `{s}`")

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources,
        "analytics_result": ana, "meta": {"route": route, "elapsed": elapsed},
    })


# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">🌻 Docs Assistant</div>
  <p class="hero-sub">Ask questions, get summaries, pull structured tables, or compare documents side by side.</p>
</div>
""", unsafe_allow_html=True)

engine_ok, engine = False, None
try:
    engine = get_engine()
    engine_ok = True
except FileNotFoundError:
    pass
except Exception as e:
    st.error(f"Couldn't start the assistant: {e}")

tab_docs, tab_chat = st.tabs(["📁 Documents", "💬 Chat"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_docs:
    col_upload, col_existing = st.columns([1, 1])

    with col_upload:
        st.markdown('<div class="sec-label">📁 Upload new documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drag & drop or click to browse",
            type=[e.lstrip(".") for e in sorted(SUPPORTED_EXT)],
            accept_multiple_files=True,
            key="doc_uploader",
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.markdown(
                f'<div style="font-size:12px;color:#C0603F;margin:4px 0 8px;font-weight:600;">'
                f'✓ {len(uploaded_files)} file(s) ready to index</div>', unsafe_allow_html=True
            )
        if st.button("✨ Index documents", disabled=not uploaded_files):
            saved, skipped = save_uploaded_files(uploaded_files, Path(config.DATA_DIR))
            if skipped:
                st.warning(f"Skipped (unsupported): {', '.join(skipped)}")
            if saved:
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("No supported files were saved.")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">⚙️ Models</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="icard"><span class="icard-lbl">LLM</span><span class="icard-val">{config.LLM_MODEL.split('/')[-1]}</span></div>
        <div class="icard"><span class="icard-lbl">Embeddings</span><span class="icard-val">{config.EMBEDDING_MODEL.split('/')[-1]}</span></div>
        """, unsafe_allow_html=True)

    with col_existing:
        st.markdown('<div class="sec-label">📋 Already indexed</div>', unsafe_allow_html=True)
        data_dir = Path(config.DATA_DIR)
        existing = sorted(data_dir.glob("*")) if data_dir.exists() else []

        if not existing:
            st.markdown(
                '<div style="font-size:13px;color:var(--tx-lo);padding:12px 0;">No documents yet — upload some on the left.</div>',
                unsafe_allow_html=True,
            )
        else:
            for fp in existing:
                icon = ICON.get(fp.suffix.lower().lstrip("."), "📄")
                size_kb = fp.stat().st_size / 1024
                st.markdown(
                    f'<div class="icard"><span class="icard-lbl">{icon} {fp.name}</span>'
                    f'<span class="icard-val">{size_kb:.1f} KB</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<hr/>", unsafe_allow_html=True)
            if st.button("🗑️ Clear entire index"):
                import shutil
                storage = Path(config.STORAGE_DIR)
                if storage.exists():
                    shutil.rmtree(storage)
                for fp in existing:
                    fp.unlink()
                st.cache_resource.clear()
                st.success("Cleared. Upload documents to start again.")
                time.sleep(0.6)
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not engine_ok:
        st.markdown("""
        <div class="empty-state">
          <div style="font-size:38px;margin-bottom:10px;">📂</div>
          <div style="font-size:18px;font-weight:700;color:var(--tx-hi);margin-bottom:6px;">No documents yet</div>
          <div style="font-size:13px;color:var(--tx-mid);">Switch to the <strong>📁 Documents</strong> tab to upload files first.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        known_files = engine.list_known_files()

        # ── Chat history (renders first, so new answers appear where the
        # conversation naturally grows -- above the controls, not below them) ──
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                ana = msg.get("analytics_result")
                if ana and ana.get("success") and ana.get("result_type") in ("dataframe", "series"):
                    st.markdown(msg["content"])
                    st.dataframe(ana["value"], use_container_width=True, hide_index=True)
                else:
                    st.markdown(msg["content"])

                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                        for s in msg["sources"]:
                            st.markdown(f"- `{s}`")

                if msg.get("meta"):
                    m = msg["meta"]
                    st.markdown(
                        f'<div class="meta-row">'
                        f'<span class="pill pill-{TASK_COLOR.get(m["route"], "coral")}">{TASK_LABEL.get(m["route"], m["route"])}</span>'
                        f'<span class="meta-dot">{m["elapsed"]:.1f}s</span></div>',
                        unsafe_allow_html=True,
                    )

        # ── Quick-action chips ────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📝  Summarize"):
                st.session_state.active_task = "summarize"
        with c2:
            if st.button("🗂️  Extract a table"):
                st.session_state.active_task = "extract"
        with c3:
            if st.button("⚖️  Compare"):
                st.session_state.active_task = "compare"

        # ── Task form: ask which doc(s) before running, same pattern for all three ──
        task = st.session_state.active_task
        if task:
            titles = {
                "summarize": ("📝 Summarize", "Pick one or more documents to summarize."),
                "extract": ("🗂️ Extract a table", "Pick the document(s) to pull data from, and describe what to extract."),
                "compare": ("⚖️ Compare documents", "Pick two or more documents to compare."),
            }
            title, sub = titles[task]

            st.markdown(f'<div class="task-card"><div class="task-card-title">{title}</div>'
                        f'<div class="task-card-sub">{sub}</div>', unsafe_allow_html=True)

            min_needed = 2 if task == "compare" else 1
            selected = st.multiselect("Documents", options=known_files, key=f"select_{task}", label_visibility="collapsed")

            extra_instructions = ""
            if task == "extract":
                extra_instructions = st.text_input(
                    "What should be extracted?", placeholder="e.g. name, date, and total for each row",
                    key="extract_instructions",
                )

            colA, colB = st.columns([1, 1])
            with colA:
                run_clicked = st.button(
                    f"Run {task}",
                    disabled=len(selected) < min_needed or (task == "extract" and not extra_instructions.strip()),
                    key=f"run_{task}",
                )
            with colB:
                st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
                cancel_clicked = st.button("Cancel", key=f"cancel_{task}")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            if cancel_clicked:
                st.session_state.active_task = None
                st.rerun()

            if run_clicked:
                st.session_state.active_task = None
                if task == "summarize":
                    question = f"Summarize {', '.join(selected)}"
                elif task == "extract":
                    question = f"{extra_instructions.strip()} (from {', '.join(selected)})"
                else:
                    question = f"Compare {', '.join(selected)}"
                run_task(question, forced_route=task, target_files=selected)
                st.rerun()

        # ── Free-form chat input (normal Q&A -- routes itself). Streamlit
        # auto-pins st.chat_input to the bottom of the viewport, so this
        # stays anchored below everything else regardless of code order. ──
        question = st.chat_input("Ask about your documents...")
        if question:
            run_task(question, forced_route=None, target_files=None)
            st.rerun()