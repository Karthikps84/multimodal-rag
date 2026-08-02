from pathlib import Path

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")

# --- NVIDIA NIM settings ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

LLM_MODEL = "meta/llama-3.1-70b-instruct"
# Local FAQ lookups don't need 70B-level reasoning -- a smaller model
# answers in ~2-4s vs ~8-15s, and is the single biggest lever for
# getting local answers under 10s end-to-end. Web/summarize/extract/
# compare (more complex, less latency-sensitive) still use LLM_MODEL.
FAST_LLM_MODEL = "meta/llama-3.1-8b-instruct"
USE_FAST_LLM_FOR_LOCAL = True
VISION_MODEL = "nvidia/neva-22b"          # used for image/figure captioning during ingest
# NOTE: baai/bge-m3 is set up on NVIDIA's side as an async/polling
# endpoint (200 fulfilled vs 202 pending, needs a status-poll GET) --
# the sync langchain wrapper doesn't handle that contract reliably
# and it 500s on every request, even trivial ones. Use a standard
# synchronous embed model instead.
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"

# Chunking: RecursiveCharacterTextSplitter measures in characters, not
# tokens -- 512 chars is only ~100-130 tokens, which fragments a 19k-token
# corpus into 260+ tiny chunks. Bigger chunks = fewer embed round-trips
# during ingest and less fragmented context per retrieved chunk.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# EMBED_BATCH_SIZE=16 was a workaround for what turned out to be a wrong
# model slug (baai/bge-m3, an async-polling endpoint incompatible with
# the sync SDK) -- not actually a batch-size problem. nv-embedqa-e5-v5 is
# a standard sync endpoint; safe to raise both for faster ingest.
EMBED_BATCH_SIZE = 48
EMBED_WORKERS   = 5            # parallel embed threads during ingest (increase if rate-limits allow)

SIMILARITY_TOP_K = 8          # over-fetch, then compress/rerank down to FINAL_TOP_K
FINAL_TOP_K = 4
SIMILARITY_THRESHOLD = 0.45   # relevance score (0-1, higher=better) below which we fall back to web

WEB_SEARCH_RESULTS = 3

# --- Multimodal / OCR ---
ENABLE_MULTIMODAL_PDF = True   # extract tables/images from PDFs
# docling is SOTA-accurate but a full neural OCR+layout pipeline --
# 10-100x slower per page than pymupdf4llm. Only worth it for scanned
# or heavily-tabled PDFs. Off by default so ingest stays fast; the
# loader still falls back to it automatically if pymupdf4llm's output
# for a given file looks too sparse (see loaders.py).
ENABLE_DOCLING_OCR = False
CAPTION_IMAGES = False         # set True to caption extracted images with VISION_MODEL (adds latency/cost)

# --- Retrieval quality ---
ENABLE_RERANK = True           # local, no extra network hop -> cheap latency-wise
USE_MMR = True                 # diversify results, avoid near-duplicate chunks
ENABLE_HYBRID_SEARCH = True    # merge BM25 + dense scores with RRF before reranking

# --- Caching ---
ANSWER_CACHE_TTL = 60          # seconds to cache identical question answers (0 = disabled)
EMBED_CACHE_SIZE = 128         # LRU slots for query-embedding cache

# --- Chat ---
MAX_HISTORY_TURNS = 6           # how many prior turns kept in context

# --- Analytics (Text-to-Pandas) ---
ENABLE_ANALYTICS   = True       # route analytical questions to pandas engine
MAX_DATAFRAME_ROWS = 500        # cap rows shown in UI table (full df still used for computation)
ANALYTICS_CHARTS   = True       # auto-render bar/line chart when result has numeric column

# --- New task types ---
ENABLE_SUMMARIZATION = True     # single/multi-doc summarization route
ENABLE_EXTRACTION = True        # structured field extraction -> table route
ENABLE_COMPARISON = True        # cross-document comparison route
SUMMARY_MAX_CHUNKS_PER_FILE = 8   # cap chunks pulled per file for summarization (latency guard)
COMPARE_MAX_CHUNKS_PER_FILE = 5   # cap chunks pulled per file for comparison (latency guard)