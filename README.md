# Local Llama RAG

Local document RAG assistant built around LangChain, Chroma, Streamlit, and NVIDIA-compatible model endpoints. The project loads documents from `multimodal-rag/data`, builds or reuses a persistent vector store in `multimodal-rag/storage`, and can fall back to web search when local context is not strong enough.

## What The Project Does

- Ingests local documents and chunks them for retrieval.
- Stores embeddings and metadata in a persistent Chroma database.
- Answers questions through a RAG pipeline with reranking and chat history handling.
- Falls back to web search when the local corpus does not provide enough signal.
- Exposes both a CLI chat loop and a Streamlit UI.

## Folder Guide

```text
multimodal-rag/
├── app.py         # Streamlit UI
├── main.py        # CLI entrypoint
├── rag_engine.py  # Core retrieval / generation pipeline
├── loaders.py     # Document loading and parsing helpers
├── web_search.py  # Search fallback and result formatting
├── config.py      # Paths, model settings, chunking, and thresholds
├── utils.py       # Token, hashing, and history helpers
├── logger.py      # Shared logging setup
├── test_api.py    # Standalone API connectivity check
├── requirements.txt
├── data/          # Source documents to index
├── storage/       # Persistent Chroma index and dataset metadata
└── logs/          # Runtime logs
```

## Current Contents

- `data/` currently contains document sources that are already wired into the index.
- `storage/` currently contains `chroma.sqlite3`, `dataset.json`, and the persisted Chroma collection directory.
- `logs/` is available for runtime output and is currently empty.

## Key Files

- [app.py](app.py) is the Streamlit interface. It builds the chat UI, initializes the RAG engine, and renders answers with sources.
- [main.py](main.py) is the CLI entrypoint. It starts the interactive terminal chat flow.
- [rag_engine.py](rag_engine.py) contains the retrieval pipeline, persistence logic, reranking, and web-search fallback orchestration.
- [loaders.py](loaders.py) handles supported document types such as PDF, DOCX, XLSX, CSV, PPTX, TXT, and MD.
- [web_search.py](web_search.py) wraps the search fallback and result formatting.
- [config.py](config.py) centralizes paths, model names, chunking settings, and retrieval thresholds.
- [utils.py](utils.py) provides token counting, dataset hashing, and chat-history trimming.
- [logger.py](logger.py) defines the shared logging format.
- [test_api.py](test_api.py) is a small connectivity script for validating the external API setup.

## Requirements

- Python 3.12+
- `NVIDIA_API_KEY`
- `SERPER_API_KEY` for web search fallback

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r multimodal-rag/requirements.txt
```

3. Create `multimodal-rag/.env` with the required keys:

```env
NVIDIA_API_KEY=your_nvidia_key
SERPER_API_KEY=your_serper_key
```

4. Add or replace source documents in `multimodal-rag/data/`.

## Run

From the repository root:

```bash
cd multimodal-rag
python main.py
```

For the Streamlit UI:

```bash
cd multimodal-rag
streamlit run app.py
```

On the first run, the app loads documents, chunks them, and builds the Chroma store under `storage/`. Later runs reuse the existing index unless the contents of `data/` change.

## Notes

- Unsupported file types in `data/` are skipped with a warning.
- `test_api.py` can be used to confirm the external API configuration independently of the RAG app.
