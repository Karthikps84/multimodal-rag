"""
==========================================================
Multi-Format Document Loader (multimodal-aware)
==========================================================
Maps file extensions to a loader and returns a flat list of
LangChain Document objects.

PDFs get special handling: pymupdf4llm extracts markdown that
preserves table structure and flags images, instead of the
plain-text flattening PyPDFLoader does. Falls back to
PyPDFLoader if pymupdf4llm isn't installed.
==========================================================
"""

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

try:
    import pymupdf4llm
    import fitz  # PyMuPDF
    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False


def _load_pdf_multimodal(path: Path) -> list[Document]:
    """
    Extract PDF page-by-page as markdown (tables preserved as
    markdown tables). Optionally caption embedded images with a
    vision model and append the caption as extra text so it's
    retrievable by the text embedder.
    """
    docs = []
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

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source_file": path.name,
                    "page": page_num,
                    "content_type": "table" if "|---" in text else "text",
                },
            )
        )
    return docs


def _caption_images(path: Path, page_num: int, images: list) -> list[str]:
    """Best-effort image captioning via NVIDIA vision NIM. Skips
    silently on any failure so ingestion never breaks over this."""
    captions = []
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        import base64
        import os

        vision_llm = ChatNVIDIA(
            model=config.VISION_MODEL,
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=config.NVIDIA_BASE_URL,
        )
        doc = fitz.open(str(path))
        page = doc[page_num]

        for img in images[:3]:  # cap per page to bound latency
            xref = img.get("xref")
            if xref is None:
                continue
            pix = fitz.Pixmap(doc, xref)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode()
            msg = vision_llm.invoke(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image/figure in one sentence."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ]
            )
            captions.append(msg.content)
        doc.close()
    except Exception as e:
        logger.warning(f"Image captioning skipped ({path.name} p.{page_num}): {e}")
    return captions


def _pdf_loader(path: Path) -> list[Document]:
    if config.ENABLE_MULTIMODAL_PDF and _HAS_PYMUPDF4LLM:
        try:
            return _load_pdf_multimodal(path)
        except Exception as e:
            logger.warning(f"Multimodal PDF extraction failed for {path.name}: {e}. Falling back.")

    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(str(path)).load()


SIMPLE_LOADER_MAP = {
    ".docx": Docx2txtLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".csv": CSVLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}

# Extensions whose native structure (rows/sheets) should NOT be
# broken up by a char-based text splitter later.
STRUCTURED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def load_documents(data_dir: Path) -> list[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found")

    files = sorted(data_dir.glob("*"))
    if not files:
        raise ValueError(f"No files found in {data_dir}")

    all_documents: list[Document] = []

    for file in files:
        ext = file.suffix.lower()

        try:
            if ext == ".pdf":
                docs = _pdf_loader(file)
            elif ext in SIMPLE_LOADER_MAP:
                logger.info(f"Loading {file.name} using {SIMPLE_LOADER_MAP[ext].__name__}")
                docs = SIMPLE_LOADER_MAP[ext](str(file)).load()
            else:
                logger.warning(f"Skipping unsupported file type: {file.name}")
                continue

            for doc in docs:
                doc.metadata.setdefault("source_file", file.name)
                doc.metadata["structured"] = ext in STRUCTURED_EXTENSIONS

            all_documents.extend(docs)

        except Exception as e:
            logger.error(f"Failed to load {file.name}: {e}")

    logger.info(f"Loaded {len(all_documents)} document(s)/chunk(s) from {len(files)} file(s)")
    return all_documents