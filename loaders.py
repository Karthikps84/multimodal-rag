"""
==========================================================
Multi-Format Document Loader  (multimodal-aware, fast)
==========================================================
PDF OCR priority chain:
  1. docling        — SOTA layout-aware OCR, table reconstruction,
                      scanned doc support via docTR neural OCR.
                      Best accuracy, but a full model pipeline --
                      10-100x slower per page. OFF by default
                      (config.ENABLE_DOCLING_OCR) to keep ingest
                      fast; auto-fallback if pymupdf4llm's output
                      looks too sparse (likely a scanned PDF).
  2. pymupdf4llm    — Fast text + markdown table extraction.
                      Zero model inference, good for digital PDFs.
                      Default primary path.
  3. PyPDFLoader    — LangChain baseline fallback.

All files in a directory are loaded in parallel using
ThreadPoolExecutor, so N PDFs load in ~1/Nth the serial time.
==========================================================
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)

from logger import logger
import config

# ── Silence docling / transformers startup spam (only matters if docling
# is actually enabled/imported) ────────────────────────────────────────────
import logging as _logging
import warnings as _warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
_logging.getLogger("transformers").setLevel(_logging.ERROR)
_logging.getLogger("docling").setLevel(_logging.WARNING)
_warnings.filterwarnings("ignore", message=".*__path__.*", category=DeprecationWarning)
_warnings.filterwarnings("ignore", message=".*Accessing.*", category=UserWarning)

_HAS_DOCLING = False
if getattr(config, "ENABLE_DOCLING_OCR", False):
    import sys as _sys
    try:
        _null = open(os.devnull, "w")
        _old_stderr, _sys.stderr = _sys.stderr, _null
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            _HAS_DOCLING = True
        finally:
            _sys.stderr = _old_stderr
            _null.close()
    except ImportError:
        _HAS_DOCLING = False

try:
    import pymupdf4llm
    import fitz  # PyMuPDF
    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False


# ── PDF loaders ───────────────────────────────────────────────────────────────

def _load_pdf_docling(path: Path) -> list[Document]:
    """Best-quality PDF loader using IBM docling (SOTA OCR + table reconstruction).
    Slow -- only invoked when ENABLE_DOCLING_OCR=True or as a sparse-output
    fallback (see _load_pdf)."""
    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = True
    pipeline_opts.do_table_structure = True
    pipeline_opts.table_structure_options.do_cell_matching = True
    pipeline_opts.ocr_options.use_gpu = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )
    result = converter.convert(str(path))
    doc_md = result.document.export_to_markdown()

    pages = doc_md.split("<!-- PageBreak -->")
    docs: list[Document] = []
    for i, page_text in enumerate(pages):
        text = page_text.strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source_file": path.name,
                "page": i,
                "content_type": "table" if "|---" in text or "| " in text else "text",
                "ocr_engine": "docling",
            },
        ))
    return docs


def _load_pdf_pymupdf(path: Path) -> list[Document]:
    """Fast layout-preserving PDF loader. Preserves markdown tables,
    inline code, headers. No OCR -- digital (text-layer) PDFs only."""
    docs: list[Document] = []
    md_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    for page in md_pages:
        text = page.get("text", "").strip()
        page_num = page.get("metadata", {}).get("page", 0)

        if config.CAPTION_IMAGES and page.get("images"):
            captions = _caption_images(path, page_num, page["images"])
            if captions:
                text += "\n\n[Image descriptions]\n" + "\n".join(captions)

        if not text:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source_file": path.name,
                "page": page_num,
                "content_type": "table" if "|---" in text else "text",
                "ocr_engine": "pymupdf4llm",
            },
        ))
    return docs


def _caption_images(path: Path, page_num: int, images: list) -> list[str]:
    captions = []
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        import base64

        vision_llm = ChatNVIDIA(
            model=config.VISION_MODEL,
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=config.NVIDIA_BASE_URL,
        )
        doc = fitz.open(str(path))
        pg = doc[page_num]
        for img in images[:3]:
            xref = img.get("xref")
            if xref is None:
                continue
            pix = fitz.Pixmap(doc, xref)
            b64 = base64.b64encode(pix.tobytes("png")).decode()
            msg = vision_llm.invoke([{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image/figure in one sentence."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }])
            captions.append(msg.content)
        doc.close()
    except Exception as e:
        logger.warning(f"Image captioning skipped ({path.name} p.{page_num}): {e}")
    return captions


def _looks_sparse(docs: list[Document], path: Path) -> bool:
    """Heuristic: if a digital-text extraction returned almost no text
    relative to page count, the PDF is probably scanned/image-based and
    needs real OCR."""
    if not docs:
        return True
    total_chars = sum(len(d.page_content) for d in docs)
    avg_per_page = total_chars / max(len(docs), 1)
    return avg_per_page < 40  # a real text page is normally hundreds+ chars


def _load_pdf(path: Path) -> list[Document]:
    """
    Fast path first: pymupdf4llm (digital PDFs, no model inference).
    Falls back to docling only if that output looks sparse (likely
    scanned) AND docling is available -- keeps the common case fast
    while still handling scanned PDFs correctly.
    """
    if getattr(config, "ENABLE_MULTIMODAL_PDF", True) and _HAS_PYMUPDF4LLM:
        try:
            logger.info(f"Loading {path.name} with pymupdf4llm")
            docs = _load_pdf_pymupdf(path)
            if not _looks_sparse(docs, path):
                return docs
            logger.warning(f"{path.name}: sparse text extraction, likely scanned")
        except Exception as e:
            logger.warning(f"pymupdf4llm failed for {path.name}: {e}; trying next tier")
            docs = []
    else:
        docs = []

    if _HAS_DOCLING:
        try:
            logger.info(f"Loading {path.name} with docling (OCR fallback)")
            docling_docs = _load_pdf_docling(path)
            if docling_docs:
                return docling_docs
        except Exception as e:
            logger.warning(f"docling failed for {path.name}: {e}; falling back")

    if docs:
        return docs

    logger.info(f"Loading {path.name} with PyPDFLoader (baseline)")
    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(str(path)).load()


# ── Simple loaders map ────────────────────────────────────────────────────────

SIMPLE_LOADER_MAP: dict[str, type] = {
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls":  UnstructuredExcelLoader,
    ".csv":  CSVLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
}

STRUCTURED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def _load_single_file(file: Path) -> list[Document]:
    ext = file.suffix.lower()
    try:
        if ext == ".pdf":
            docs = _load_pdf(file)
        elif ext in SIMPLE_LOADER_MAP:
            logger.info(f"Loading {file.name} using {SIMPLE_LOADER_MAP[ext].__name__}")
            docs = SIMPLE_LOADER_MAP[ext](str(file)).load()
        else:
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []

        for doc in docs:
            doc.metadata.setdefault("source_file", file.name)
            doc.metadata["structured"] = ext in STRUCTURED_EXTENSIONS

        return docs

    except Exception as e:
        logger.error(f"Failed to load {file.name}: {e}")
        return []


def load_documents(data_dir: Path, max_workers: int | None = None) -> list[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found")

    files = sorted(f for f in data_dir.glob("*") if f.is_file())
    if not files:
        raise ValueError(f"No files found in {data_dir}")

    supported_exts = set(SIMPLE_LOADER_MAP.keys()) | {".pdf"}
    files = [f for f in files if f.suffix.lower() in supported_exts]
    if not files:
        raise FileNotFoundError(f"No supported documents found in {data_dir}")

    workers = max_workers or min(8, (os.cpu_count() or 4))
    all_documents: list[Document] = []

    if len(files) == 1:
        all_documents = _load_single_file(files[0])
    else:
        logger.info(f"Loading {len(files)} files in parallel (workers={workers})...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_load_single_file, f): f for f in files}
            for future in as_completed(futures):
                fname = futures[future].name
                try:
                    all_documents.extend(future.result())
                except Exception as e:
                    logger.error(f"Parallel load failed for {fname}: {e}")

    logger.info(f"Loaded {len(all_documents)} document(s)/chunk(s) from {len(files)} file(s)")
    return all_documents