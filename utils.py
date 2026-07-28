import hashlib
from pathlib import Path

import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(encoding.encode(text))


def pretty_number(number: int) -> str:
    return "{:,}".format(number)


def compute_dataset_hash(data_dir: Path) -> str:
    """
    Cheap fingerprint (name + size + mtime) instead of hashing file
    contents -- keeps startup fast even on large corpora. Good enough
    to detect add/remove/edit for local single-user use.
    """
    sha = hashlib.sha256()
    for file in sorted(data_dir.glob("*")):
        stat = file.stat()
        sha.update(file.name.encode())
        sha.update(str(stat.st_size).encode())
        sha.update(str(int(stat.st_mtime)).encode())
    return sha.hexdigest()


def trim_history(history: list, max_turns: int) -> list:
    """Keep only the last N (user, assistant) turns."""
    return history[-max_turns:] if max_turns > 0 else []