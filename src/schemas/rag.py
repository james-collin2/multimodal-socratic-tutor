"""
src/schemas/rag.py

Pydantic v2 schemas for RAG pipeline — chunks, retrieval results, citations.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from src.schemas.base import BaseSchema, IdentifiedSchema


class DocumentChunk(IdentifiedSchema):
    """A single text chunk from the corpus after splitting."""

    content: str = Field(min_length=1)
    source: str  # e.g. "openStax/chapter_14.pdf"
    page: int | None = None
    chapter: str | None = None
    topic_tags: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseSchema):
    """A chunk returned by the retriever with relevance score."""

    chunk: DocumentChunk
    score: float = Field(ge=0.0, le=1.0)


class RetrievalResult(BaseSchema):
    """Full result from a single RAG retrieval call."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    assembled_context: str = ""
    citations: list[str] = Field(default_factory=list)

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0

    @property
    def top_score(self) -> float:
        if not self.chunks:
            return 0.0
        return max(c.score for c in self.chunks)


class ImageMetadata(BaseSchema):
    """Metadata for an anatomical diagram image."""

    filename: str
    source: str
    license: str
    structures: list[str] = Field(default_factory=list)
    region: str
    difficulty: str = "intermediate"  # beginner | intermediate | advanced
    ot_relevance: str | None = None
