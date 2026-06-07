"""
src/utils/helpers.py

Shared pure utility functions — no business logic, no external dependencies.
All functions are stateless and independently testable.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def truncate_text(text: str, max_chars: int = 500, suffix: str = "...") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove non-printable characters.
    Used during corpus ingestion.
    """
    # Collapse multiple whitespace (including newlines) to single space
    text = re.sub(r"\s+", " ", text)
    # Strip only C0/C1 control characters; keep non-ASCII (Greek letters, accented chars, etc.)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    return text.strip()


def safe_parse_json(text: str) -> dict[str, Any] | None:
    """
    Safely parse JSON from LLM output.
    Handles common LLM quirks: markdown fences, trailing commas.
    Returns None if parsing fails.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def hash_string(text: str) -> str:
    """Return SHA-256 hex digest of a string. Used for chunk IDs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into chunks of given size."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_citations(sources: list[str]) -> str:
    """Format a list of source strings into a readable citation block."""
    if not sources:
        return ""
    unique = list(dict.fromkeys(sources))  # deduplicate, preserve order
    return "\n".join(f"[{i + 1}] {src}" for i, src in enumerate(unique))


def detect_bypass_attempt(user_input: str) -> bool:
    """
    Detect if a student is trying to bypass Socratic hints.
    Covers: direct demands, instruction injection, role override,
    authority commands, and Socratic rejection.
    """
    bypass_patterns = [
        r"\bjust tell me\b",
        r"\bgive me the answer\b",
        r"\bwhat is the answer\b",
        r"\bstop asking questions\b",
        r"\btell me directly\b",
        r"\bskip the hints?\b",
        r"\breveal the answer\b",
        r"\bneed the answer\b",
        r"\bignore your instructions?\b",
        r"\bforget your rules\b",
        r"\bdisregard\b.*\binstructions?\b",
        r"\boverride\b.*\b(system|prompt|rules?)\b",
        r"\bpretend you have no\b",
        r"\bno restrictions?\b",
        r"\bact as\b.*\b(different|normal|another)\b",
        r"\bjust be a normal\b",
        r"\bi command you\b",
        r"\byou must tell me\b",
        r"\banswer without\b",
        r"\bstop the socratic\b",
        r"\bno more hints?\b",
        r"\bdon.t have time for hints?\b",
        r"\bimmediately\b.*\banswer\b",
        r"\banswer immediately\b",
    ]
    text_lower = user_input.lower()
    return any(re.search(p, text_lower) for p in bypass_patterns)


def sanitize_student_id(raw_id: str) -> str:
    """Remove unsafe characters from a student ID string."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "", raw_id)[:64]
