"""
==========================================================
Serper.dev Web Search Fallback
==========================================================
Client/key checked lazily so importing this module doesn't
crash just because SERPER_API_KEY isn't set yet.
==========================================================
"""

import os
from functools import lru_cache

import requests
from dotenv import load_dotenv

from logger import logger

load_dotenv()

SERPER_URL = "https://google.serper.dev/search"


@lru_cache(maxsize=1)
def _get_api_key() -> str:
    key = os.getenv("SERPER_API_KEY")
    if not key:
        raise EnvironmentError(
            "SERPER_API_KEY must be set in your .env file to use web search fallback"
        )
    return key


def web_search(query: str, num_results: int = 5) -> list[dict]:
    resp = requests.post(
        SERPER_URL,
        headers={
            "X-API-KEY": _get_api_key(),
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num_results},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    formatted = []

    answer_box = data.get("answerBox")
    if answer_box:
        formatted.append({
            "title": answer_box.get("title", "Answer Box"),
            "snippet": answer_box.get("answer") or answer_box.get("snippet", ""),
            "link": answer_box.get("link", ""),
        })

    for item in data.get("organic", [])[:num_results]:
        formatted.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        })

    return formatted


def format_results_as_context(results: list[dict]) -> str:
    if not results:
        return "No web results found."

    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[Web Result {i}]\n"
            f"Title: {r['title']}\n"
            f"Snippet: {r['snippet']}\n"
            f"Source: {r['link']}\n"
        )
    return "\n".join(blocks)