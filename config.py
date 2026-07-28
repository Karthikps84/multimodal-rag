from pathlib import Path

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")

# --- NVIDIA NIM settings ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

LLM_MODEL = "meta/llama-3.1-70b-instruct"
VISION_MODEL = "nvidia/neva-22b"          # used for image/figure captioning during ingest
# NOTE: baai/bge-m3 is set up on NVIDIA's side as an async/polling
# endpoint (200 fulfilled vs 202 pending, needs a status-poll GET) --
# the sync langchain wrapper doesn't handle that contract reliably
# and it 500s on every request, even trivial ones. Use a standard
# synchronous embed model instead.
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

EMBED_BATCH_SIZE = 16          # keep small -- NIM embed endpoint 500s on oversized batches

SIMILARITY_TOP_K = 8          # over-fetch, then compress/rerank down to FINAL_TOP_K
FINAL_TOP_K = 4
SIMILARITY_THRESHOLD = 0.45   # relevance score (0-1, higher=better) below which we fall back to web

WEB_SEARCH_RESULTS = 3

# --- Multimodal ---
ENABLE_MULTIMODAL_PDF = True   # extract tables/images from PDFs via pymupdf4llm
CAPTION_IMAGES = False         # set True to caption extracted images with VISION_MODEL (adds latency/cost)

# --- Retrieval quality ---
ENABLE_RERANK = True           # local, no extra network hop -> cheap latency-wise
USE_MMR = True                 # diversify results, avoid near-duplicate chunks

# --- Chat ---
MAX_HISTORY_TURNS = 6           # how many prior turns kept in context