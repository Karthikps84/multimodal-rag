import os
import json
import time
from pathlib import Path
from typing import TypedDict, Optional

from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


class GraphState(TypedDict):
    question: str
    history: list
    docs: list
    max_score: float
    answer: str
    sources: list
    used_web: bool


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
            timeout=120,   # mistral-medium can be slow, esp. first call after cold start
        )

        self.embeddings = NVIDIAEmbeddings(
            model=embedding_model,
            api_key=NVIDIA_API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            truncate="END",
            # NIM embed endpoints 500 on batches that are too large
            # (too many texts, or combined tokens too high). Small
            # batch = more requests but reliable.
            max_batch_size=getattr(config, "EMBED_BATCH_SIZE", 16),
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        self.vectorstore: Optional[Chroma] = None
        self.history: list = []  # list of (question, answer)

        self.reranker = None
        if config.ENABLE_RERANK and _HAS_FLASHRANK:
            # Local cross-encoder reranker -- no network hop, adds
            # ~tens of ms, not a real latency concern.
            self.reranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
        elif config.ENABLE_RERANK:
            logger.warning("flashrank not installed -- reranking disabled. `pip install flashrank` to enable.")

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

        self.parser = StrOutputParser()
        self.local_chain = self.rag_prompt | self.llm | self.parser
        self.web_chain = self.web_prompt | self.llm | self.parser

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
    # Build / load vector store
    ##############################################################

    def _build_vectorstore(self):
        logger.info("Loading documents...")
        documents = load_documents(self.data_dir)

        total_tokens = sum(token_count(d.page_content) for d in documents)
        logger.info(f"Total documents: {len(documents)} | Total tokens: {total_tokens:,}")

        logger.info("Splitting into chunks (structured files kept intact)...")
        to_split = [d for d in documents if not d.metadata.get("structured")]
        keep_whole = [d for d in documents if d.metadata.get("structured")]
        chunks = self.splitter.split_documents(to_split) + keep_whole
        chunks = self._sanitize_chunks(chunks)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Building Chroma vector store...")
        start = time.time()

        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.storage_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._add_documents_with_retry(chunks)

        logger.info(f"Vector store built in {time.time() - start:.2f} sec")
        self._save_hash()

    @staticmethod
    def _sanitize_chunks(chunks: list) -> list:
        """
        Clean chunk text before embedding. PDF math/formula regions
        (common in academic papers) often extract as garbled Unicode
        -- private-use-area glyphs, stray control chars, broken font
        codepoints -- which NIM's embed endpoint can 500 on. Strip
        that out, normalize, and drop chunks that are mostly garbage
        after cleaning (rather than sending garbage and eating a 500).
        """
        import re
        import unicodedata

        # control chars (keep \n \t), and Unicode Private Use Area
        # (U+E000-F8FF) where broken math/symbol fonts often land
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

            # heuristic: if >30% of remaining chars are non-printable
            # or replacement-ish after cleaning, this chunk is likely
            # still garbled extraction -- skip it rather than risk a 500
            printable = sum(1 for ch in text if ch.isprintable())
            if printable / max(len(text), 1) < 0.7:
                logger.warning(f"Dropping likely-garbled chunk from {c.metadata.get('source_file')} (p.{c.metadata.get('page','-')})")
                dropped += 1
                continue

            c.page_content = text
            clean.append(c)

        if dropped:
            logger.info(f"Sanitization dropped {dropped} empty/garbled chunk(s)")
        return clean

    def _add_documents_with_retry(self, chunks: list, batch_size: int = None):
        """
        Adds chunks to Chroma in small batches, retrying with an
        even smaller batch (and a short backoff) if the NIM embed
        endpoint 500s -- rather than failing the whole ingest run
        over one bad batch.
        """
        batch_size = batch_size or getattr(config, "EMBED_BATCH_SIZE", 16)
        i = 0
        while i < len(chunks):
            batch = chunks[i : i + batch_size]
            try:
                self.vectorstore.add_documents(batch)
                i += batch_size
            except Exception as e:
                if batch_size <= 1:
                    bad = batch[0]
                    preview = repr(bad.page_content[:150])
                    logger.error(
                        f"Skipping unembeddable chunk from {bad.metadata.get('source_file')} "
                        f"(p.{bad.metadata.get('page','-')}): {e}\n  content preview: {preview}"
                    )
                    i += 1
                    continue
                logger.warning(f"Embed batch failed (size={batch_size}): {e}. Retrying smaller.")
                batch_size = max(1, batch_size // 2)
                time.sleep(1)

    def _load_vectorstore(self):
        logger.info("Loading existing Chroma vector store...")
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(self.storage_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def initialize(self):
        self.storage_dir.mkdir(exist_ok=True)
        self._validate_data_folder()
        has_existing = self.storage_dir.exists() and any(self.storage_dir.iterdir())

        if has_existing and not self._dataset_changed():
            logger.info("Dataset unchanged. Loading existing index.")
            self._load_vectorstore()
        else:
            logger.info("Dataset changed or no index found. Rebuilding.")
            self._build_vectorstore()

    def _validate_data_folder(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.data_dir}. "
                f"Create it and upload documents before running ingestion."
            )

        files = [p for p in self.data_dir.iterdir() if p.is_file()]
        if not files:
            raise FileNotFoundError(
                f"No files found in {self.data_dir}. "
                "Please upload documents to the data folder, then run again."
            )

        supported = [p for p in files if p.suffix.lower() in self.SUPPORTED_EXTENSIONS]
        if not supported:
            supported_list = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise FileNotFoundError(
                f"No supported documents found in {self.data_dir}. "
                f"Upload files with one of: {supported_list}."
            )

    ##############################################################
    # Retrieval (MMR + rerank), using LangChain's built-in
    # relevance-score conversion instead of hand-rolled math
    ##############################################################

    def _retrieve(self, question: str):
        if config.USE_MMR:
            docs = self.vectorstore.max_marginal_relevance_search(
                question, k=self.similarity_top_k, fetch_k=self.similarity_top_k * 3
            )
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                question, k=self.similarity_top_k
            )
            score_map = {d.page_content: s for d, s in scored}
            max_score = max(score_map.values(), default=0.0)
        else:
            scored = self.vectorstore.similarity_search_with_relevance_scores(
                question, k=self.similarity_top_k
            )
            docs = [d for d, _ in scored]
            max_score = max((s for _, s in scored), default=0.0)

        if self.reranker and docs:
            docs = self._rerank(question, docs)

        docs = docs[: self.final_top_k]

        for d in docs:
            logger.info(f"Retrieved [{d.metadata.get('source_file')}]: {d.page_content[:80]}...")

        return docs, max_score

    def _rerank(self, question: str, docs: list) -> list:
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
        results = self.reranker.rerank(RerankRequest(query=question, passages=passages))
        order = [r["id"] for r in results]
        return [docs[i] for i in order]

    ##############################################################
    # LangGraph nodes
    ##############################################################

    def _format_history(self, history: list) -> str:
        if not history:
            return "(none)"
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in history)

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
            "context": context,
            "question": state["question"],
            "history": self._format_history(state["history"]),
        })
        logger.info(f"Local generation time: {time.time() - start:.2f}s")
        state["answer"] = answer
        state["sources"] = sorted({d.metadata.get("source_file", "unknown") for d in state["docs"]})
        state["used_web"] = False
        return state

    def _node_web_generate(self, state: GraphState) -> GraphState:
        start = time.time()
        results = web_search(state["question"], num_results=config.WEB_SEARCH_RESULTS)
        logger.info(f"Web search time: {time.time() - start:.2f}s | results={len(results)}")

        context = format_results_as_context(results)
        start = time.time()
        answer = self.web_chain.invoke({
            "context": context,
            "question": state["question"],
            "history": self._format_history(state["history"]),
        })
        logger.info(f"Web generation time: {time.time() - start:.2f}s")

        state["answer"] = answer
        state["sources"] = [f"{r['title']} ({r['link']})" for r in results]
        state["used_web"] = True
        return state

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("local_generate", self._node_local_generate)
        graph.add_node("web_generate", self._node_web_generate)

        graph.set_entry_point("retrieve")
        graph.add_conditional_edges(
            "retrieve", self._route, {"local": "local_generate", "web": "web_generate"}
        )
        graph.add_edge("local_generate", END)
        graph.add_edge("web_generate", END)

        return graph.compile()

    ##############################################################
    # Public API
    ##############################################################

    def ask(self, question: str) -> dict:
        result = self.graph.invoke({
            "question": question,
            "history": self.history,
            "docs": [],
            "max_score": 0.0,
            "answer": "",
            "sources": [],
            "used_web": False,
        })

        self.history.append((question, result["answer"]))
        self.history = trim_history(self.history, config.MAX_HISTORY_TURNS)

        return result

    ##############################################################
    # Interactive chat
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
