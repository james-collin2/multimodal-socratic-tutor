"""
src/core/rag/embedder.py

Embedder — converts text chunks into embedding vectors.
Single responsibility: text list in → vector list out.

Wraps the EmbeddingModel singleton from models layer.
Handles batching to avoid OOM on large corpora.
"""

from __future__ import annotations

from src.models.embedding_model import get_embedding_model
from src.schemas.rag import DocumentChunk
from src.utils.exceptions import EmbeddingError
from src.utils.helpers import chunk_list
from src.utils.logger import logger


class Embedder:
    """
    Async embedder for DocumentChunk objects.

    Args:
        batch_size: Number of chunks to embed per batch (default 32)
    """

    def __init__(self, batch_size: int = 32) -> None:
        self._batch_size = batch_size
        self._model = get_embedding_model()

    async def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[list[float]]:
        """
        Embed a list of DocumentChunks in batches.

        Args:
            chunks: List of DocumentChunk objects to embed

        Returns:
            List of embedding vectors (one per chunk)

        Raises:
            EmbeddingError: If embedding fails
        """
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        batches = chunk_list(texts, self._batch_size)
        all_embeddings: list[list[float]] = []

        logger.info(
            "Embedding {n} chunks in {b} batches of {size}",
            n=len(chunks),
            b=len(batches),
            size=self._batch_size,
        )

        for i, batch in enumerate(batches):
            try:
                embeddings = await self._model.encode(batch)
                all_embeddings.extend(embeddings)
                logger.debug(
                    "Embedded batch {i}/{total}",
                    i=i + 1,
                    total=len(batches),
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to embed batch {i + 1}/{len(batches)}",
                    detail=str(e),
                ) from e

        logger.info(
            "Embedding complete: {n} vectors, dim={dim}",
            n=len(all_embeddings),
            dim=len(all_embeddings[0]) if all_embeddings else 0,
        )
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        Args:
            query: The search query text

        Returns:
            Embedding vector for the query
        """
        try:
            return await self._model.encode_single(query)
        except Exception as e:
            raise EmbeddingError(
                "Failed to embed query",
                detail=str(e),
            ) from e

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (loads model if needed)."""
        self._model._load()
        return 384  # all-MiniLM-L6-v2 default
