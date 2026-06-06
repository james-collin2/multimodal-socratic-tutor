"""
src/core/rag/chunker.py

Chunker — splits raw text into overlapping chunks with metadata.
Single responsibility: text in → DocumentChunk list out.

Uses LangChain RecursiveCharacterTextSplitter internally
but wraps it behind a clean interface so the rest of the
codebase never imports LangChain directly.
"""

from __future__ import annotations

from uuid import uuid4

from src.schemas.rag import DocumentChunk
from src.utils.helpers import clean_text, hash_string
from src.utils.logger import logger


class Chunker:
    """
    Splits text documents into overlapping chunks.

    Args:
        chunk_size:    Target token count per chunk (default 512)
        chunk_overlap: Overlap tokens between chunks (default 64)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = self._build_splitter()

    def _build_splitter(self) -> object:
        """Build LangChain text splitter — isolated here."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size * 4,  # approx chars per token
                chunk_overlap=self._chunk_overlap * 4,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        except ImportError:
            logger.warning("langchain_text_splitters not found — using simple splitter")
            return None

    def _simple_split(self, text: str) -> list[str]:
        """Fallback splitter when LangChain is unavailable."""
        chars_per_chunk = self._chunk_size * 4
        overlap_chars = self._chunk_overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chars_per_chunk, len(text))
            chunks.append(text[start:end])
            start += chars_per_chunk - overlap_chars
        return chunks

    def chunk_text(
        self,
        text: str,
        source: str,
        topic_tags: list[str] | None = None,
        chapter: str | None = None,
    ) -> list[DocumentChunk]:
        """
        Split text into DocumentChunk objects.

        Args:
            text:       Raw text to split
            source:     Source identifier (e.g. "openStax/chapter_14.pdf")
            topic_tags: List of topic tags for filtering
            chapter:    Chapter name if applicable

        Returns:
            List of DocumentChunk objects ready for embedding
        """
        cleaned = clean_text(text)
        if not cleaned:
            logger.warning("Empty text after cleaning for source: {s}", s=source)
            return []

        # Split text
        if self._splitter is not None:
            raw_chunks = self._splitter.split_text(cleaned)
        else:
            raw_chunks = self._simple_split(cleaned)

        # Build DocumentChunk objects
        chunks: list[DocumentChunk] = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 50:  # skip tiny fragments
                continue

            chunk = DocumentChunk(
                id=uuid4(),
                content=chunk_text,
                source=source,
                chapter=chapter,
                topic_tags=topic_tags or [],
                token_count=len(chunk_text) // 4,  # rough estimate
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "chunk_id": hash_string(chunk_text),
                },
            )
            chunks.append(chunk)

        logger.debug(
            "Chunked {source}: {n} chunks from {chars} chars",
            source=source,
            n=len(chunks),
            chars=len(cleaned),
        )
        return chunks

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap
