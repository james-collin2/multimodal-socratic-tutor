"""
src/utils/exceptions.py

All custom exceptions for socratOT.
Raise specific exceptions internally — only catch at the app/UI boundary.

Usage:
    from src.utils.exceptions import LLMUnavailableError
    raise LLMUnavailableError("OpenAI API key not set")
"""

from __future__ import annotations

# ── Base ──────────────────────────────────────────────────────────────────────


class SocratOTError(Exception):
    """Base exception for all socratOT errors."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} — {self.detail}"
        return self.message


# ── LLM / Model ───────────────────────────────────────────────────────────────


class LLMUnavailableError(SocratOTError):
    """Raised when the OpenAI API is unreachable or the key is missing."""


class LLMTimeoutError(SocratOTError):
    """Raised when an LLM call exceeds the configured timeout."""


class LLMResponseParseError(SocratOTError):
    """Raised when LLM returns malformed JSON or unexpected format."""


class ModelNotFoundError(SocratOTError):
    """Raised when a requested model is not available from the provider."""


# ── RAG ───────────────────────────────────────────────────────────────────────


class VectorStoreError(SocratOTError):
    """Raised when ChromaDB or FAISS operations fail."""


class EmbeddingError(SocratOTError):
    """Raised when embedding model fails to encode text."""


class CorpusEmptyError(SocratOTError):
    """Raised when the vector store has no documents indexed."""


class RetrievalError(SocratOTError):
    """Raised when semantic retrieval fails."""


# ── Multimodal ────────────────────────────────────────────────────────────────


class ImageProcessingError(SocratOTError):
    """Raised when image upload or pre-processing fails."""


class ImageAnalysisError(SocratOTError):
    """Raised when the vision model fails to analyse an image."""


class UnsupportedImageFormatError(SocratOTError):
    """Raised when uploaded file is not a supported image format."""


# ── Conversation ──────────────────────────────────────────────────────────────


class SessionNotFoundError(SocratOTError):
    """Raised when a session_id does not exist in the database."""


class SessionExpiredError(SocratOTError):
    """Raised when a session has passed its expiry window."""


class InvalidPhaseTransitionError(SocratOTError):
    """Raised when trying to transition to an invalid conversation phase."""


# ── Persistence ───────────────────────────────────────────────────────────────


class DatabaseError(SocratOTError):
    """Raised when a SQLite operation fails."""


class ConfigurationError(SocratOTError):
    """Raised when required config is missing or invalid."""
