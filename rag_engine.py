import os
import json
import time
import hashlib
import uuid
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict, Optional, Iterator, Callable

from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LCDoc

from langgraph.graph import StateGraph, END

from loaders import load_documents
from web_search import web_search, format_results_as_context
from utils import token_count, compute_dataset_hash, trim_history
from logger import logger

import config

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

try:
    from flashrank import Ranker, RerankRequest
    _HAS_FLASHRANK = True
except ImportError:
    _HAS_FLASHRANK = False

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    from cachetools import TTLCache
    _HAS_CACHETOOLS = True
except ImportError:
    _HAS_CACHETOOLS = False

try:
    from analytics import AnalyticsEngine
    _HAS_ANALYTICS_MODULE = True
except ImportError:
    _HAS_ANALYTICS_MODULE = False


ProgressFn = Optional[Callable[[int, str], None]]


class GraphState(TypedDict):
    question: str
    history: list
    docs: list
    max_score: float
    answer: str
    sources: list
    used_web: bool
    route_type: str                    # "local" | "web" | "analytics" | "summarize" | "extract" | "compare"
    analytics_result: Optional[dict]   # populated on analytics/extract routes (table-shaped results)
    target_files: list                 # file(s) the task is scoped to, if identifiable from the question
    forced_route: Optional[str]        # set by the UI when the user explicitly picked a task + files,
                                        # bypassing the keyword guess entirely


# ---------------------------------------------------------------------------
# Module-level helpers (no self dependency)
# ---------------------------------------------------------------------------

def _content_key(doc) -> str:
    """Stable dedup/fusion key -- full-content hash, not a truncated
    prefix (which risks collisions between chunks sharing boilerplate
    headers)."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _mmr_filter(scored_docs: list, k: int, lambda_mult: float = 0.5) -> list:
    """Lightweight local MMR filter -- avoids a second Chroma network call."""
    if not scored_docs:
        return []

    def _sim(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        union = ta | tb
        return len(ta & tb) / len(union) if union else 0.0

    selected: list = []
    remaining = list(scored_docs)

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:
            selected_texts = [d.page_content for d in selected]

            def _mmr_score(item, _texts=selected_texts):
                doc, rel = item
                max_sim = max(_sim(doc.page_content, t) for t in _texts)
                return lambda_mult * rel - (1 - lambda_mult) * max_sim

            best = max(remaining, key=_mmr_score)
        selected.append(best[0])
        remaining.remove(best)

    return selected


def _reciprocal_rank_fusion(dense_scored: list, bm25_docs: list, k: int = 60) -> list:
    """Merge dense + BM25 rankings via Reciprocal Rank Fusion. Keyed on a
    full-content hash (not a truncated prefix) to avoid merging distinct
    chunks that happen to share a common prefix."""
    scores: dict = {}
    doc_map: dict = {}

    for rank, (doc, _) in enumerate(dense_scored):
        key = _content_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = _content_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


class RAGEngine:

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".docx", ".xlsx", ".xls", ".csv", ".pptx", ".txt", ".md"
    }

    ##############################################################
    # Constructor
    ##############################################################

    def __init__(
        self,
        data_dir=config.DATA_DIR,
        storage_dir=config.STORAGE_DIR,
        llm_model=config.LLM_MODEL,
        embedding_model=config.EMBEDDING_MODEL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        similarity_top_k=config.SIMILARITY_TOP_K,
        final_top_k=config.FINAL_TOP_K,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
    ):
        self.data_dir = Path(data_dir)
        self.storage_dir = Path(storage_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.final_top_k = final_top_k
        self.similarity_threshold = similarity_threshold

        if not NVIDIA_API_KEY:
            raise EnvironmentError("NVIDIA_API_KEY must be set in your .env file")

        logger.info("Initializing models via NVIDIA NIM")
        logger.info(f"LLM: {llm_model} | Embeddings: {embedding_model}")

        self.llm = ChatNVIDIA(
            model=llm_model,
            api_key=NVIDIA_API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            temperature=0.1,
            timeout=120,
        )

        # Separate, smaller/faster model for local-knowledge-base answers --
        # this is the single biggest lever for keeping local answers under
        # 10s; 70B-class models routinely take 8-15s+ per call, an 8B model
        # answers in 2-4s and is plenty for FAQ-style lookups grounded in
        # retrieved context.
        if getattr(config, "USE_FAST_LLM_FOR_LOCAL", True) and getattr(config, "FAST_LLM_MODEL", None):
            self.fast_llm = ChatNVIDIA(
                model=config.FAST_LLM_MODEL,
                api_key=NVIDIA_API_KEY,
                base_url=config.NVIDIA_BASE_URL,
                temperature=0.1,
                timeout=60,
            )
        else:
            self.fast_llm = self.llm

        self.embeddings = NVIDIAEmbeddings(
            model=embedding_model,
            api_key=NVIDIA_API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            truncate="END",
            max_batch_size=getattr(config, "EMBED_BATCH_SIZE", 16),
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self.vectorstore: Optional[Chroma] = None
        self.history: list = []

        self._bm25 = None
        self._bm25_corpus: list = []

        self.reranker = None
        if config.ENABLE_RERANK and _HAS_FLASHRANK:
            self.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
        elif config.ENABLE_RERANK:
            logger.warning("flashrank not installed -- reranking disabled. `pip install flashrank`.")

        if getattr(config, "ENABLE_HYBRID_SEARCH", True) and not _HAS_BM25:
            logger.warning("rank-bm25 not installed -- hybrid search disabled. `pip install rank-bm25`.")

        ttl = getattr(config, "ANSWER_CACHE_TTL", 60)
        if ttl > 0 and _HAS_CACHETOOLS:
            self._answer_cache = TTLCache(maxsize=64, ttl=ttl)
        else:
            self._answer_cache = None
            if ttl > 0 and not _HAS_CACHETOOLS:
                logger.warning("cachetools not installed -- answer cache disabled. `pip install cachetools`.")

        embed_cache_size = getattr(config, "EMBED_CACHE_SIZE", 128)
        _embed_fn = self.embeddings.embed_query

        @lru_cache(maxsize=embed_cache_size)
        def _cached_embed(text: str) -> list:
            return _embed_fn(text)

        self._cached_embed = _cached_embed

        # ── Prompts & chains ──────────────────────────────────────────────────
        self.rag_prompt = ChatPromptTemplate.from_template(
            "You are a helpful FAQ assistant. Answer the question using ONLY "
            "the context below. Cite the source file (and page, if given) for "
            "each claim like [source_file p.X]. If the answer isn't in the "
            "context, say you don't know rather than guessing.\n\n"
            "Conversation so far:\n{history}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        self.web_prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the user's question using "
            "ONLY the web search results below. Note these results may be "
            "time-sensitive. If the results don't contain the answer, say so.\n\n"
            "Conversation so far:\n{history}\n\n"
            "Web Search Results:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        self.summarize_prompt = ChatPromptTemplate.from_template(
            "Summarize the document content below. Be concise but complete: "
            "cover the main points, key findings/arguments, and any important "
            "numbers or conclusions. If multiple files are included, summarize "
            "each one under its own heading, then add a short combined takeaway.\n\n"
            "User's request: {question}\n\n"
            "Document content:\n{context}\n\n"
            "Summary:"
        )

        self.extract_prompt = ChatPromptTemplate.from_template(
            "Extract the requested information from the content below and "
            "return it as a markdown table ONLY -- no prose before or after, "
            "just the table. Pick sensible column headers based on what the "
            "user asked for. If a field isn't present for a row, use an empty "
            "cell rather than guessing.\n\n"
            "User's request: {question}\n\n"
            "Content:\n{context}\n\n"
            "Markdown table:"
        )

        self.compare_prompt = ChatPromptTemplate.from_template(
            "Compare the documents below based on the user's request. Return "
            "a markdown comparison table where rows are the aspects being "
            "compared and columns are the documents, followed by a short "
            "prose summary of the key differences.\n\n"
            "User's request: {question}\n\n"
            "Documents:\n{context}\n\n"
            "Comparison:"
        )

        self.parser = StrOutputParser()
        self.local_chain = self.rag_prompt | self.fast_llm | self.parser
        self.web_chain = self.web_prompt | self.llm | self.parser
        # Same reasoning as local_chain above -- these tasks don't need
        # 70B-level reasoning, and were the actual cause of >1min task
        # latency (large multi-chunk context + slow model = compounding).
        self.summarize_chain = self.summarize_prompt | self.fast_llm | self.parser
        self.extract_chain = self.extract_prompt | self.fast_llm | self.parser
        self.compare_chain = self.compare_prompt | self.fast_llm | self.parser

        # ── Analytics engine (optional -- module not always present) ─────────
        self.analytics = None
        if _HAS_ANALYTICS_MODULE and getattr(config, "ENABLE_ANALYTICS", True):
            try:
                self.analytics = AnalyticsEngine(data_dir=self.data_dir, llm=self.fast_llm)
            except Exception as e:
                logger.warning(f"AnalyticsEngine init failed, analytics route disabled: {e}")

        self.graph = self._build_graph()

    ##############################################################
    # Dataset hash helpers
    ##############################################################

    def _hash_file(self):
        return self.storage_dir / "dataset.json"

    def _save_hash(self):
        with open(self._hash_file(), "w") as f:
            json.dump({"dataset_hash": compute_dataset_hash(self.data_dir)}, f, indent=4)

    def _dataset_changed(self):
        hash_file = self._hash_file()
        if not hash_file.exists():
            return True
        with open(hash_file) as f:
            old_hash = json.load(f)["dataset_hash"]
        return old_hash != compute_dataset_hash(self.data_dir)

    ##############################################################
    # Build / load vector store  (now with real progress reporting)
    ##############################################################

    def _build_vectorstore(self, on_progress: ProgressFn = None):
        def report(pct, msg):
            logger.info(msg)
            if on_progress:
                on_progress(pct, msg)

        report(2, "Loading documents...")
        documents = load_documents(self.data_dir)

        total_tokens = sum(token_count(d.page_content) for d in documents)
        report(8, f"Loaded {len(documents)} document(s), {total_tokens:,} tokens")

        report(12, "Splitting into chunks...")
        to_split = [d for d in documents if not d.metadata.get("structured")]
        keep_whole = [d for d in documents if d.metadata.get("structured")]
        chunks = self.splitter.split_documents(to_split) + keep_whole
        chunks = self._sanitize_chunks(chunks)
        report(18, f"Created {len(chunks)} chunks")

        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.storage_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

        start = time.time()
        self._add_documents_with_retry(chunks, on_progress=on_progress)
        report(92, f"Vector store built in {time.time() - start:.2f}s")

        self._save_hash()
        self._build_bm25_index(chunks)
        report(100, "Index ready")

    @staticmethod
    def _sanitize_chunks(chunks: list) -> list:
        import re
        import unicodedata

        _CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
        _PUA_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")

        clean = []
        dropped = 0
        for c in chunks:
            text = c.page_content or ""
            text = unicodedata.normalize("NFKC", text)
            text = _CONTROL_RE.sub("", text)
            text = _PUA_RE.sub(" ", text)
            text = text.strip()

            if not text:
                dropped += 1
                continue

            printable = sum(1 for ch in text if ch.isprintable())
            if printable / max(len(text), 1) < 0.7:
                logger.warning(
                    f"Dropping likely-garbled chunk from "
                    f"{c.metadata.get('source_file')} (p.{c.metadata.get('page','-')})"
                )
                dropped += 1
                continue

            c.page_content = text
            clean.append(c)

        if dropped:
            logger.info(f"Sanitization dropped {dropped} empty/garbled chunk(s)")
        return clean

    def _add_documents_with_retry(self, chunks: list, batch_size: int = None, on_progress: ProgressFn = None):
        """
        Parallel batch embedding + serial Chroma insert. Chunk IDs use
        uuid4 (stable-format, collision-free) instead of a truncated
        content hash. Reports real progress per completed batch.
        """
        import concurrent.futures

        batch_size = batch_size or getattr(config, "EMBED_BATCH_SIZE", 16)
        embed_workers = getattr(config, "EMBED_WORKERS", 3)

        batches = [chunks[i: i + batch_size] for i in range(0, len(chunks), batch_size)]
        logger.info(
            f"Embedding {len(chunks)} chunks in {len(batches)} batches "
            f"(batch_size={batch_size}, workers={embed_workers})..."
        )

        def _embed_batch_safe(batch: list, size_hint: int):
            current_size = size_hint
            while current_size >= 1:
                sub_batches = [batch[j: j + current_size] for j in range(0, len(batch), current_size)]
                try:
                    results = []
                    for sb in sub_batches:
                        texts = [d.page_content for d in sb]
                        vectors = self.embeddings.embed_documents(texts)
                        results.append((sb, vectors))
                    return results
                except Exception as e:
                    if current_size <= 1:
                        logger.error(f"Skipping unembeddable chunk: {e}")
                        return None
                    logger.warning(f"Embed batch failed (size={current_size}): {e}. Retrying smaller.")
                    current_size = max(1, current_size // 2)
                    time.sleep(1)
            return None

        batch_results: list = [None] * len(batches)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=embed_workers) as pool:
            future_to_idx = {
                pool.submit(_embed_batch_safe, batch, batch_size): idx
                for idx, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    batch_results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch {idx} embed worker raised: {e}")
                completed += 1
                if on_progress and batches:
                    pct = 20 + int(70 * completed / len(batches))
                    on_progress(pct, f"Embedding batch {completed}/{len(batches)}...")

        for result in batch_results:
            if result is None:
                continue
            for sub_batch, vectors in result:
                ids = [str(uuid.uuid4()) for _ in sub_batch]
                try:
                    self.vectorstore._collection.add(
                        ids=ids,
                        embeddings=vectors,
                        documents=[d.page_content for d in sub_batch],
                        metadatas=[d.metadata for d in sub_batch],
                    )
                except Exception:
                    try:
                        self.vectorstore.add_documents(sub_batch, ids=ids)
                    except Exception as e2:
                        logger.error(f"Insert fallback also failed: {e2}")

    def _load_vectorstore(self, on_progress: ProgressFn = None):
        logger.info("Loading existing Chroma vector store...")
        if on_progress:
            on_progress(30, "Loading existing index...")
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.storage_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._rebuild_bm25_from_vectorstore()
        if on_progress:
            on_progress(100, "Index ready")

    def initialize(self, on_progress: ProgressFn = None):
        self.storage_dir.mkdir(exist_ok=True)
        self._validate_data_folder()
        has_existing = self.storage_dir.exists() and any(self.storage_dir.iterdir())

        if has_existing and not self._dataset_changed():
            logger.info("Dataset unchanged. Loading existing index.")
            self._load_vectorstore(on_progress=on_progress)
        else:
            logger.info("Dataset changed or no index found. Rebuilding.")
            self._build_vectorstore(on_progress=on_progress)

    def _validate_data_folder(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.data_dir}. Create it and upload documents first."
            )
        files = [p for p in self.data_dir.iterdir() if p.is_file()]
        if not files:
            raise FileNotFoundError(f"No files found in {self.data_dir}. Upload documents first.")
        supported = [p for p in files if p.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        if not supported:
            supported_list = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise FileNotFoundError(f"No supported documents in {self.data_dir}. Use: {supported_list}.")

    ##############################################################
    # BM25 index management
    ##############################################################

    def _build_bm25_index(self, chunks: list):
        if not _HAS_BM25 or not getattr(config, "ENABLE_HYBRID_SEARCH", True):
            return
        logger.info(f"Building BM25 index over {len(chunks)} chunks...")
        self._bm25_corpus = chunks
        tokenized = [c.page_content.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index ready.")

    def _rebuild_bm25_from_vectorstore(self):
        if not _HAS_BM25 or not getattr(config, "ENABLE_HYBRID_SEARCH", True):
            return
        try:
            logger.info("Rebuilding BM25 index from stored Chroma documents...")
            raw = self.vectorstore.get(include=["documents", "metadatas"])
            docs = [
                LCDoc(page_content=text, metadata=meta)
                for text, meta in zip(raw["documents"], raw["metadatas"])
            ]
            self._build_bm25_index(docs)
        except Exception as e:
            logger.warning(f"BM25 index rebuild failed (hybrid search disabled): {e}")

    ##############################################################
    # Retrieval
    ##############################################################

    def _retrieve(self, question: str, source_filter: list = None):
        t_start = time.time()

        where = {"source_file": {"$in": source_filter}} if source_filter else None
        t0 = time.time()
        scored = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=self.similarity_top_k, filter=where
        )
        max_score = max((s for _, s in scored), default=0.0)
        logger.info(f"  [retrieve] embed+ANN search: {time.time() - t0:.3f}s | top_score={max_score:.4f}")

        t0 = time.time()
        if self._bm25 and self._bm25_corpus and getattr(config, "ENABLE_HYBRID_SEARCH", True) and not source_filter:
            tokens = question.lower().split()
            bm25_scores_arr = self._bm25.get_scores(tokens)
            top_bm25_indices = sorted(
                range(len(bm25_scores_arr)), key=lambda i: bm25_scores_arr[i], reverse=True
            )[: self.similarity_top_k]
            bm25_docs = [self._bm25_corpus[i] for i in top_bm25_indices]
            docs = _reciprocal_rank_fusion(scored, bm25_docs)
            logger.info(f"  [retrieve] BM25 + RRF fusion: {time.time() - t0:.3f}s")
        else:
            if getattr(config, "USE_MMR", True):
                docs = _mmr_filter(scored, k=self.similarity_top_k)
            else:
                docs = [d for d, _ in scored]
            logger.info(f"  [retrieve] MMR/passthrough: {time.time() - t0:.3f}s")

        t0 = time.time()
        if self.reranker and docs:
            docs = self._rerank(question, docs)
        logger.info(f"  [retrieve] rerank: {time.time() - t0:.3f}s")

        docs = docs[: self.final_top_k]
        for d in docs:
            logger.info(f"Retrieved [{d.metadata.get('source_file')}]: {d.page_content[:80]}...")

        logger.info(f"  [retrieve] TOTAL: {time.time() - t_start:.3f}s")
        return docs, max_score

    def _rerank(self, question: str, docs: list) -> list:
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results = self.reranker.rerank(RerankRequest(query=question, passages=passages))
        order = [r["id"] for r in results]
        return [docs[i] for i in order]

    ##############################################################
    # File-name resolution for task-scoped routes (summarize/extract/compare)
    ##############################################################

    def list_known_files(self) -> list:
        """Public accessor for the UI (e.g. a file-selection multiselect)."""
        return self._list_known_files()

    def _list_known_files(self) -> list:
        if not self.data_dir.exists():
            return []
        return sorted(p.name for p in self.data_dir.glob("*") if p.is_file())

    def _match_files_in_question(self, question: str) -> list:
        q = question.lower()
        matches = []
        for fname in self._list_known_files():
            stem = Path(fname).stem.lower()
            if stem in q or fname.lower() in q:
                matches.append(fname)
        return matches

    def _fetch_file_chunks(self, filename: str, max_chunks: int) -> list:
        """Cheap metadata-filtered fetch from Chroma -- no embedding call."""
        try:
            raw = self.vectorstore.get(where={"source_file": filename}, include=["documents", "metadatas"])
            docs = [
                LCDoc(page_content=text, metadata=meta)
                for text, meta in zip(raw["documents"], raw["metadatas"])
            ]
            docs.sort(key=lambda d: d.metadata.get("page", 0) if isinstance(d.metadata.get("page"), int) else 0)
            return docs[:max_chunks]
        except Exception as e:
            logger.warning(f"Could not fetch chunks for {filename}: {e}")
            return []

    ##############################################################
    # LangGraph nodes
    ##############################################################

    def _format_history(self, history: list) -> str:
        if not history:
            return "(none)"
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in history)

    _ANALYTICAL_HINTS = (
        "average", "avg", "mean", "median", "sum", "total", "count", "how many",
        "percent", "percentage", "correlation", "trend", "maximum", "minimum",
        "max ", "min ", "highest", "lowest", "growth", "ratio", "distribution",
        "chart", "plot", "graph of", "group by", "per year", "per month",
    )

    def _node_intent_classify(self, state: GraphState) -> GraphState:
        """
        Fast keyword-based intent routing -- deliberately NOT an extra
        LLM call for the common case, to avoid adding latency to every
        single question. Order matters: summarize/extract/compare
        keywords are checked before the generic analytics/retrieve
        fallback. The analytics path is additionally gated behind a
        cheap keyword prefilter before we let it call its own LLM-based
        classifier (self.analytics.is_analytical) -- without that gate,
        EVERY question pays for an extra LLM round-trip just to decide
        it isn't analytical.
        """
        t0 = time.time()

        if state.get("forced_route"):
            state["route_type"] = state["forced_route"]
            if not state.get("target_files"):
                state["target_files"] = self._match_files_in_question(state["question"])
            logger.info(f"Intent (forced): {state['route_type']} | {time.time()-t0:.3f}s")
            return state

        q = state["question"].lower()
        state["target_files"] = self._match_files_in_question(state["question"])

        if getattr(config, "ENABLE_SUMMARIZATION", True) and any(
            kw in q for kw in ("summarize", "summarise", "summary", "tl;dr", "tldr", "overview of", "recap")
        ):
            state["route_type"] = "summarize"
        elif getattr(config, "ENABLE_EXTRACTION", True) and any(
            kw in q for kw in ("extract", "pull out", "list all", "table of", "tabulate")
        ):
            state["route_type"] = "extract"
        elif getattr(config, "ENABLE_COMPARISON", True) and any(
            kw in q for kw in ("compare", "comparison", "difference between", " vs ", " versus ")
        ):
            state["route_type"] = "compare"
        elif (
            self.analytics is not None
            and getattr(self.analytics, "dataframes", None)
            and any(kw in q for kw in self._ANALYTICAL_HINTS)
            and self.analytics.is_analytical(state["question"])
        ):
            state["route_type"] = "analytics"
        else:
            state["route_type"] = "retrieve"

        logger.info(f"Intent classify: {state['route_type']} | {time.time()-t0:.3f}s")
        return state

    def _route_from_intent(self, state: GraphState) -> str:
        return state["route_type"]

    def _node_analytics(self, state: GraphState) -> GraphState:
        result = self.analytics.run(state["question"])
        state["analytics_result"] = result
        state["answer"] = result["answer"]
        state["sources"] = [result.get("filename", "")] if result.get("filename") else []
        state["used_web"] = False
        state["max_score"] = 1.0
        return state

    def _resolve_target_files(self, state: GraphState, min_files: int = 1) -> list:
        """
        Decide which file(s) a summarize/extract/compare request applies to:
        1. Files explicitly named in the question.
        2. If none named and only one file exists in the corpus, use it.
        3. Otherwise fall back to the top distinct source files from a
           quick relevance search (still one cheap dense retrieval call).
        """
        files = state.get("target_files") or []
        if files:
            return files

        known = self._list_known_files()
        if len(known) == 1:
            return known

        docs, _ = self._retrieve(state["question"])
        seen = []
        for d in docs:
            f = d.metadata.get("source_file")
            if f and f not in seen:
                seen.append(f)
            if len(seen) >= max(min_files, 2):
                break
        return seen or known[:min_files]

    def _node_summarize(self, state: GraphState) -> GraphState:
        files = self._resolve_target_files(state, min_files=1)
        max_per_file = getattr(config, "SUMMARY_MAX_CHUNKS_PER_FILE", 12)

        parts = []
        for f in files:
            chunks = self._fetch_file_chunks(f, max_per_file)
            if not chunks:
                continue
            body = "\n\n".join(c.page_content for c in chunks)
            parts.append(f"=== {f} ===\n{body}")

        context = "\n\n".join(parts) if parts else "(no content found for the requested file(s))"

        start = time.time()
        answer = self.summarize_chain.invoke({"context": context, "question": state["question"]})
        logger.info(f"Summarize generation time: {time.time() - start:.2f}s")

        state["answer"] = answer
        state["sources"] = files
        state["used_web"] = False
        state["max_score"] = 1.0
        return state

    def _node_extract(self, state: GraphState) -> GraphState:
        files = self._resolve_target_files(state, min_files=1)

        if files:
            max_per_file = getattr(config, "SUMMARY_MAX_CHUNKS_PER_FILE", 12)
            parts = []
            for f in files:
                chunks = self._fetch_file_chunks(f, max_per_file)
                body = "\n\n".join(c.page_content for c in chunks)
                parts.append(f"=== {f} ===\n{body}")
            context = "\n\n".join(parts)
        else:
            docs, _ = self._retrieve(state["question"])
            context = "\n\n".join(d.page_content for d in docs)
            files = sorted({d.metadata.get("source_file", "unknown") for d in docs})

        start = time.time()
        table_md = self.extract_chain.invoke({"context": context, "question": state["question"]})
        logger.info(f"Extraction generation time: {time.time() - start:.2f}s")

        parsed = self._markdown_table_to_dataframe(table_md)
        if parsed is not None:
            state["analytics_result"] = {
                "success": True,
                "result_type": "dataframe",
                "value": parsed,
                "filename": ", ".join(files),
                "generated_code": None,
            }
            state["answer"] = f"Extracted {len(parsed)} row(s) from {', '.join(files)}."
        else:
            # Fall back to showing the raw markdown table as text
            state["answer"] = table_md
            state["analytics_result"] = None

        state["sources"] = files
        state["used_web"] = False
        state["max_score"] = 1.0
        return state

    @staticmethod
    def _markdown_table_to_dataframe(md_text: str):
        """Best-effort parse of an LLM-produced markdown table into a
        pandas DataFrame, for a nicer UI than raw text. Returns None if
        the text doesn't look like a table."""
        try:
            import pandas as pd
            import re

            lines = [l for l in md_text.strip().splitlines() if l.strip().startswith("|")]
            if len(lines) < 2:
                return None

            def split_row(line):
                return [c.strip() for c in line.strip().strip("|").split("|")]

            header = split_row(lines[0])
            # skip the separator row (---|---|---)
            data_lines = [l for l in lines[1:] if not re.match(r"^\|?[\s:\-|]+\|?$", l)]
            rows = [split_row(l) for l in data_lines]
            rows = [r for r in rows if len(r) == len(header)]
            if not rows:
                return None
            return pd.DataFrame(rows, columns=header)
        except Exception:
            return None

    def _node_compare(self, state: GraphState) -> GraphState:
        files = self._resolve_target_files(state, min_files=2)
        max_per_file = getattr(config, "COMPARE_MAX_CHUNKS_PER_FILE", 6)

        parts = []
        for f in files:
            chunks = self._fetch_file_chunks(f, max_per_file)
            if not chunks:
                continue
            body = "\n\n".join(c.page_content for c in chunks)
            parts.append(f"=== {f} ===\n{body}")

        context = "\n\n".join(parts) if parts else "(not enough distinct files found to compare)"

        start = time.time()
        answer = self.compare_chain.invoke({"context": context, "question": state["question"]})
        logger.info(f"Compare generation time: {time.time() - start:.2f}s")

        state["answer"] = answer
        state["sources"] = files
        state["used_web"] = False
        state["max_score"] = 1.0
        return state

    def _node_retrieve(self, state: GraphState) -> GraphState:
        docs, max_score = self._retrieve(state["question"])
        state["docs"] = docs
        state["max_score"] = max_score
        return state

    def _route(self, state: GraphState) -> str:
        if state["max_score"] >= self.similarity_threshold:
            logger.info(f"Routing -> LOCAL (score={state['max_score']:.4f})")
            return "local"
        logger.info(f"Routing -> WEB (score={state['max_score']:.4f})")
        return "web"

    def _node_local_generate(self, state: GraphState) -> GraphState:
        context = "\n\n".join(
            f"[{d.metadata.get('source_file')} p.{d.metadata.get('page', '-')}] {d.page_content}"
            for d in state["docs"]
        )
        start = time.time()
        answer = self.local_chain.invoke({
            "context": context, "question": state["question"], "history": self._format_history(state["history"]),
        })
        logger.info(f"Local generation time: {time.time() - start:.2f}s")
        state["answer"] = answer
        state["sources"] = sorted({d.metadata.get("source_file", "unknown") for d in state["docs"]})
        state["used_web"] = False
        state["route_type"] = "local"
        return state

    def _node_web_generate(self, state: GraphState) -> GraphState:
        question = state["question"]
        history_str = self._format_history(state["history"])

        with ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(web_search, question, config.WEB_SEARCH_RESULTS).result()

        context = format_results_as_context(results)
        start = time.time()
        answer = self.web_chain.invoke({"context": context, "question": question, "history": history_str})
        logger.info(f"Web generation time: {time.time() - start:.2f}s")

        state["answer"] = answer
        state["sources"] = [f"{r['title']} ({r['link']})" for r in results]
        state["used_web"] = True
        state["route_type"] = "web"
        return state

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("intent_classify", self._node_intent_classify)
        graph.add_node("analytics", self._node_analytics)
        graph.add_node("summarize", self._node_summarize)
        graph.add_node("extract", self._node_extract)
        graph.add_node("compare", self._node_compare)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("local_generate", self._node_local_generate)
        graph.add_node("web_generate", self._node_web_generate)

        graph.set_entry_point("intent_classify")

        graph.add_conditional_edges(
            "intent_classify",
            self._route_from_intent,
            {
                "analytics": "analytics",
                "summarize": "summarize",
                "extract": "extract",
                "compare": "compare",
                "retrieve": "retrieve",
            },
        )

        graph.add_edge("analytics", END)
        graph.add_edge("summarize", END)
        graph.add_edge("extract", END)
        graph.add_edge("compare", END)

        graph.add_conditional_edges(
            "retrieve", self._route, {"local": "local_generate", "web": "web_generate"},
        )
        graph.add_edge("local_generate", END)
        graph.add_edge("web_generate", END)

        return graph.compile()

    ##############################################################
    # Public API
    ##############################################################

    def prepare_stream(self, question: str, target_files: list = None, forced_route: str = None) -> dict:
        """
        Resolves routing + retrieval/context WITHOUT calling the LLM, so the
        caller (the UI) can stream the generation chain's tokens directly
        instead of waiting for the full answer. Covers local/web/summarize/
        compare -- extract and analytics return streamable=False since they
        need the complete output before it can be parsed into a table.

        Returns a dict with at least {"streamable": bool}. When streamable,
        also includes {"chain", "inputs", "route", "sources", "max_score"}.
        Call finalize() after consuming the stream to commit history/cache.
        """
        state: GraphState = {
            "question": question, "history": self.history, "docs": [], "max_score": 0.0,
            "answer": "", "sources": [], "used_web": False, "route_type": "",
            "analytics_result": None, "target_files": target_files or [],
            "forced_route": forced_route,
        }

        if self._answer_cache is not None:
            cache_key = f"{forced_route}|{','.join(target_files or [])}|{question.strip().lower()}"
            if cache_key in self._answer_cache:
                cached = self._answer_cache[cache_key]
                return {"streamable": False, "cached": True, "result": cached}

        state = self._node_intent_classify(state)
        route = state["route_type"]
        history_str = self._format_history(self.history)

        if route == "summarize":
            files = self._resolve_target_files(state, min_files=1)
            max_per_file = getattr(config, "SUMMARY_MAX_CHUNKS_PER_FILE", 8)
            parts = []
            for f in files:
                chunks = self._fetch_file_chunks(f, max_per_file)
                if chunks:
                    parts.append(f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks))
            context = "\n\n".join(parts) if parts else "(no content found for the requested file(s))"
            return {
                "streamable": True, "route": "summarize", "sources": files, "max_score": 1.0,
                "chain": self.summarize_chain, "inputs": {"context": context, "question": question},
            }

        if route == "compare":
            files = self._resolve_target_files(state, min_files=2)
            max_per_file = getattr(config, "COMPARE_MAX_CHUNKS_PER_FILE", 5)
            parts = []
            for f in files:
                chunks = self._fetch_file_chunks(f, max_per_file)
                if chunks:
                    parts.append(f"=== {f} ===\n" + "\n\n".join(c.page_content for c in chunks))
            context = "\n\n".join(parts) if parts else "(not enough distinct files found to compare)"
            return {
                "streamable": True, "route": "compare", "sources": files, "max_score": 1.0,
                "chain": self.compare_chain, "inputs": {"context": context, "question": question},
            }

        if route in ("extract", "analytics"):
            return {"streamable": False, "route": route}

        # route == "retrieve" -> resolve local vs web
        docs, max_score = self._retrieve(question)
        if max_score >= self.similarity_threshold:
            context = "\n\n".join(
                f"[{d.metadata.get('source_file')} p.{d.metadata.get('page', '-')}] {d.page_content}"
                for d in docs
            )
            sources = sorted({d.metadata.get("source_file", "unknown") for d in docs})
            return {
                "streamable": True, "route": "local", "sources": sources, "max_score": max_score,
                "chain": self.local_chain, "inputs": {"context": context, "question": question, "history": history_str},
            }

        results = web_search(question, config.WEB_SEARCH_RESULTS)
        context = format_results_as_context(results)
        sources = [f"{r['title']} ({r['link']})" for r in results]
        return {
            "streamable": True, "route": "web", "sources": sources, "max_score": max_score,
            "chain": self.web_chain, "inputs": {"context": context, "question": question, "history": history_str},
        }

    def finalize(self, question: str, answer: str, route: str, sources: list, target_files: list = None, forced_route: str = None):
        """Commit a streamed answer to history/cache -- call after the UI
        has finished consuming prepare_stream()'s chain.stream() output."""
        self.history.append((question, answer))
        self.history = trim_history(self.history, config.MAX_HISTORY_TURNS)

        if self._answer_cache is not None:
            cache_key = f"{forced_route}|{','.join(target_files or [])}|{question.strip().lower()}"
            self._answer_cache[cache_key] = {
                "answer": answer, "sources": sources, "route_type": route,
                "used_web": route == "web", "analytics_result": None, "max_score": 1.0,
            }

    def ask(self, question: str, target_files: list = None, forced_route: str = None) -> dict:
        """
        Blocking path -- used for extract/analytics (need full output before
        parsing) and as a fallback. target_files / forced_route let the
        caller (e.g. the UI, after the user picked "Summarize" + files) skip
        the keyword-based intent guess entirely and go straight to the task.
        """
        cache_key = f"{forced_route}|{','.join(target_files or [])}|{question.strip().lower()}"
        if self._answer_cache is not None and cache_key in self._answer_cache:
            logger.info("Answer cache HIT.")
            return self._answer_cache[cache_key]

        t_total = time.time()
        result = self.graph.invoke({
            "question": question, "history": self.history, "docs": [], "max_score": 0.0,
            "answer": "", "sources": [], "used_web": False, "route_type": "",
            "analytics_result": None, "target_files": target_files or [],
            "forced_route": forced_route,
        })
        logger.info(f"===== ask() TOTAL: {time.time() - t_total:.3f}s | route={result.get('route_type')} =====")

        self.history.append((question, result["answer"]))
        self.history = trim_history(self.history, config.MAX_HISTORY_TURNS)

        if self._answer_cache is not None:
            self._answer_cache[cache_key] = result

        return result

    ##############################################################
    # Interactive chat (CLI)
    ##############################################################

    def chat(self):
        print("=" * 80)
        print("Multimodal RAG Assistant (LangChain + LangGraph + NVIDIA NIM) — type 'exit' to quit")
        print("=" * 80)

        while True:
            question = input("\nYou : ").strip()
            if question.lower() in ["exit", "quit"]:
                break
            if not question:
                continue

            result = self.ask(question)
            label = "(from web)" if result["used_web"] else ""
            print(f"\nAnswer {label}:\n{result['answer']}\n")
            print("Sources:")
            for s in result["sources"]:
                print(f" - {s}")
            print()

        print("\nGoodbye!")