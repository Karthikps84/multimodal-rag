"""
==========================================================
Main Entry Point
==========================================================
Initializes the LangGraph-orchestrated RAG engine (NVIDIA NIM
LLM + embeddings, Chroma vector store, Serper web fallback)
and starts the interactive chat loop.

Author: Karthik
==========================================================
"""

import sys
import requests

from rag_engine import RAGEngine
from logger import logger


def main():
    try:
        logger.info("Starting Multimodal RAG Assistant...")

        rag = RAGEngine()
        rag.initialize()
        rag.chat()

    except requests.exceptions.Timeout:
        logger.error("Request to NVIDIA NIM timed out.")
        logger.error("The model may be under heavy load or cold-starting -- try again, "
                      "or raise the timeout in rag_engine.py (ChatNVIDIA(timeout=...)).")
        sys.exit(1)

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Network error reaching NVIDIA NIM: {e}")
        logger.error("Check your internet connection and that api.nvidia.com is reachable.")
        sys.exit(1)

    except EnvironmentError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Check your .env file for NVIDIA_API_KEY and SERPER_API_KEY.")
        sys.exit(1)

    except FileNotFoundError as e:
        logger.error(f"Data error: {e}")
        logger.error("Make sure your data/ folder exists and contains documents.")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()