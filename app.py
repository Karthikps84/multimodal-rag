"""
==========================================================
Streamlit UI for the Multimodal RAG Assistant
==========================================================
Run locally:   streamlit run app.py
Deploy free:   push to GitHub -> share.streamlit.io -> point
               at this file. Add NVIDIA_API_KEY and
               SERPER_API_KEY as "Secrets" in the dashboard
               (same syntax as .env, no need to commit .env).
==========================================================
"""

import time
from pathlib import Path
import streamlit as st

from rag_engine import RAGEngine
from web_search import web_search, format_results_as_context
import config

st.set_page_config(page_title="RAG Assistant", page_icon="🔎", layout="wide")

SUPPORTED_UPLOAD_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".xls", ".csv", ".pptx", ".txt", ".md"
}


@st.cache_resource(show_spinner="Loading models and vector index...")
def get_engine() -> RAGEngine:
    rag = RAGEngine()
    rag.initialize()
    return rag


def stream_local_answer(engine: RAGEngine, question: str, history: list, docs: list):
    """Stream tokens from the local-generation chain so the UI fills in
    progressively instead of blocking for the full response."""
    context = "\n\n".join(
        f"[{d.metadata.get('source_file')} p.{d.metadata.get('page', '-')}] {d.page_content}"
        for d in docs
    )
    hist_str = engine._format_history(history)
    for chunk in engine.local_chain.stream({
        "context": context, "question": question, "history": hist_str
    }):
        yield chunk


def stream_web_answer(engine: RAGEngine, question: str, history: list, results: list):
    context = format_results_as_context(results)
    hist_str = engine._format_history(history)
    for chunk in engine.web_chain.stream({
        "context": context, "question": question, "history": hist_str
    }):
        yield chunk


def save_uploaded_files(uploaded_files: list, data_dir: Path) -> tuple[int, list[str]]:
    """Persist uploaded files into data_dir and return save stats."""
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    skipped_files: list[str] = []

    for file_obj in uploaded_files:
        suffix = Path(file_obj.name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            skipped_files.append(file_obj.name)
            continue

        destination = data_dir / file_obj.name
        destination.write_bytes(file_obj.getbuffer())
        saved_count += 1

    return saved_count, skipped_files


st.title("🔎 Multimodal RAG Assistant")
st.caption("LangChain + LangGraph + NVIDIA NIM — supports LaTeX equations, tables, and cited sources")

if "upload_notice" in st.session_state:
    st.success(st.session_state.pop("upload_notice"))

with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Add files to the data folder",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_UPLOAD_EXTENSIONS)],
        accept_multiple_files=True,
    )

    if st.button("Save And Process Uploaded Files", disabled=not uploaded_files):
        saved_count, skipped_files = save_uploaded_files(uploaded_files, Path(config.DATA_DIR))

        if skipped_files:
            st.warning("Skipped unsupported files: " + ", ".join(skipped_files))

        if saved_count > 0:
            st.session_state["upload_notice"] = (
                f"Saved {saved_count} file(s) to {config.DATA_DIR}. Rebuilding index."
            )
            st.cache_resource.clear()
            st.rerun()
        else:
            st.info("No supported files were saved.")

    st.subheader("Settings")
    st.write(f"**LLM:** {config.LLM_MODEL}")
    st.write(f"**Embeddings:** {config.EMBEDDING_MODEL}")
    st.write(f"**Local score threshold:** {config.SIMILARITY_THRESHOLD}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

try:
    engine = get_engine()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Upload your documents into the data folder (supported: .pdf, .docx, .xlsx, .xls, .csv, .pptx, .txt, .md), then refresh."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"}]
if "history" not in st.session_state:
    st.session_state.history = []   # [(q, a)] fed into engine prompts

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.time()
        docs, max_score = engine._retrieve(question)

        placeholder = st.empty()
        full_answer = ""

        if max_score >= engine.similarity_threshold:
            route_label = "📚 local knowledge base"
            for chunk in stream_local_answer(engine, question, st.session_state.history, docs):
                full_answer += chunk
                placeholder.markdown(full_answer + " ▌")
            sources = sorted({d.metadata.get("source_file", "unknown") for d in docs})
        else:
            route_label = "🌐 web search"
            results = web_search(question, num_results=config.WEB_SEARCH_RESULTS)
            for chunk in stream_web_answer(engine, question, st.session_state.history, results):
                full_answer += chunk
                placeholder.markdown(full_answer + " ▌")
            sources = [f"{r['title']} ({r['link']})" for r in results]

        placeholder.markdown(full_answer)
        elapsed = time.time() - start

        st.caption(f"Answered from {route_label} · score={max_score:.2f} · {elapsed:.1f}s")
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"- {s}")

    st.session_state.history.append((question, full_answer))
    st.session_state.history = st.session_state.history[-config.MAX_HISTORY_TURNS:]
    st.session_state.messages.append({
        "role": "assistant", "content": full_answer, "sources": sources
    })