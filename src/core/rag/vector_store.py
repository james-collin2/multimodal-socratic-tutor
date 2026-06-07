"""
src/core/rag/vector_store.py

VectorStore — abstract base + ChromaDB and FAISS implementations.
Single responsibility: store and retrieve embedding vectors.

Factory function get_vector_store() returns correct implementation
based on VECTOR_STORE_TYPE in settings. No caller needs to know
which backend is used.
"""

from __future__ import annotations

import asyncio
import pickle
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from src.config.settings import VectorStoreType, get_settings
from src.core.rag.embedder import Embedder
from src.schemas.rag import DocumentChunk, RetrievedChunk
from src.utils.exceptions import CorpusEmptyError, VectorStoreError
from src.utils.logger import logger

# ── Abstract base ─────────────────────────────────────────────────────────────


class VectorStore(ABC):
    """Abstract vector store — all implementations must follow this contract."""

    @abstractmethod
    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and store a list of DocumentChunks."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Search for top_k most similar chunks to query."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return number of stored chunks."""
        ...

    @abstractmethod
    async def reset(self) -> None:
        """Clear all stored chunks."""
        ...


# ── ChromaDB implementation ───────────────────────────────────────────────────


def _make_noop_embedding_function() -> object:
    """
    A no-op Chroma embedding function.

    We compute all embeddings ourselves via the OpenAI ``Embedder`` and always
    pass ``embeddings`` / ``query_embeddings`` to Chroma explicitly, so Chroma
    never needs to embed anything. Supplying this avoids Chroma falling back to
    its built-in ONNX model (``ONNXMiniLM_L6_V2``), which would require the
    heavy ``onnxruntime`` package to be installed just to open a collection.
    """
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class _NoOpEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            raise RuntimeError(
                "Embeddings are computed externally (OpenAI) — pass "
                "embeddings/query_embeddings to Chroma explicitly."
            )

    return _NoOpEmbeddingFunction()


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-backed vector store with persistent local storage.
    Primary vector store for socratOT.
    """

    COLLECTION_NAME = "socratot_anatomy"

    def __init__(self) -> None:
        settings = get_settings()
        self._persist_dir = str(settings.chroma_persist_path)
        self._embedder = Embedder()
        self._client: object = None
        self._collection: object = None

    def _get_collection(self) -> object:
        """Lazily initialise ChromaDB client and collection."""
        if self._collection is not None:
            return self._collection

        try:
            # Silence ChromaDB's PostHog telemetry. Even with
            # anonymized_telemetry=False it logs a harmless "Failed to send
            # telemetry event" (posthog version mismatch) — suppress at the source.
            import logging
            import os

            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
            logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=_make_noop_embedding_function(),
            )
            logger.info(
                "ChromaDB collection ready: {name} at {path}",
                name=self.COLLECTION_NAME,
                path=self._persist_dir,
            )
            return self._collection

        except ImportError as e:
            raise VectorStoreError(
                "chromadb is not installed",
                detail=str(e),
            ) from e
        except Exception as e:
            raise VectorStoreError(
                "Failed to initialise ChromaDB",
                detail=str(e),
            ) from e

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed chunks and add to ChromaDB collection."""
        if not chunks:
            return

        collection = self._get_collection()

        # Embed all chunks
        embeddings = await self._embedder.embed_chunks(chunks)

        # Prepare ChromaDB inputs
        ids = [str(chunk.id) for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "chapter": chunk.chapter or "",
                "topic_tags": ",".join(chunk.topic_tags),
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            ),
        )
        logger.info("Added {n} chunks to ChromaDB", n=len(chunks))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Semantic search — returns top_k most similar chunks."""
        collection = self._get_collection()

        count = await self.count()
        if count == 0:
            raise CorpusEmptyError(
                "Vector store is empty",
                detail="Run scripts/ingest_corpus.py first",
            )

        query_embedding = await self._embedder.embed_query(query)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            ),
        )

        retrieved: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for _doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance: 0=identical, 2=opposite
            # Convert to similarity score 0-1
            score = max(0.0, 1.0 - (dist / 2.0))

            if score < min_score:
                continue

            chunk = DocumentChunk(
                content=doc,
                source=meta.get("source", ""),
                chapter=meta.get("chapter") or None,
                topic_tags=meta.get("topic_tags", "").split(","),
                token_count=int(meta.get("token_count", 0)),
            )

            retrieved.append(RetrievedChunk(chunk=chunk, score=round(score, 4)))

        logger.debug(
            "ChromaDB search: query={q!r}, results={n}",
            q=query[:50],
            n=len(retrieved),
        )
        return retrieved

    async def count(self) -> int:
        """Return number of stored chunks."""
        try:
            collection = self._get_collection()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, collection.count)
        except Exception:
            return 0

    async def reset(self) -> None:
        """Delete and recreate the collection."""
        if self._client is not None:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
        logger.warning("ChromaDB collection reset — all chunks deleted")


# ── FAISS implementation ──────────────────────────────────────────────────────


class FAISSVectorStore(VectorStore):
    """
    FAISS-backed vector store for benchmark comparison.
    In-memory with optional disk persistence.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._index_path = Path(settings.faiss_index_path)
        self._embedder = Embedder()
        self._index: object = None
        self._chunks: list[DocumentChunk] = []

    def _get_index(self, dim: int = 384) -> object:
        """Lazily initialise FAISS index."""
        if self._index is not None:
            return self._index

        # Try loading from disk
        if self._index_path.exists():
            try:
                import pickle

                import faiss

                self._index = faiss.read_index(str(self._index_path))
                chunks_path = self._index_path.with_suffix(".pkl")
                if chunks_path.exists():
                    with open(chunks_path, "rb") as f:
                        # trusted: loads the index cache this app wrote itself
                        self._chunks = pickle.load(f)  # noqa: S301
                logger.info("Loaded FAISS index from disk")
                return self._index
            except Exception as e:
                logger.warning("Could not load FAISS index: {e}", e=str(e))

        try:
            import faiss

            # Inner product index (use with normalised vectors for cosine similarity)
            self._index = faiss.IndexFlatIP(dim)
            logger.info("Created new FAISS index (dim={d})", d=dim)
            return self._index
        except ImportError as e:
            raise VectorStoreError("faiss-cpu is not installed", detail=str(e)) from e

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        import faiss
        import numpy as np

        embeddings = await self._embedder.embed_chunks(chunks)
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalise for cosine similarity
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = self._get_index(dim)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: index.add(vectors))
        self._chunks.extend(chunks)

        # Persist to disk
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        await loop.run_in_executor(None, lambda: faiss.write_index(index, str(self._index_path)))
        chunks_path = self._index_path.with_suffix(".pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info("Added {n} chunks to FAISS index", n=len(chunks))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        import faiss
        import numpy as np

        if not self._chunks:
            raise CorpusEmptyError("FAISS index is empty")

        query_vec = await self._embedder.embed_query(query)
        q = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(q)

        index = self._get_index()
        k = min(top_k, len(self._chunks))

        loop = asyncio.get_running_loop()
        scores, indices = await loop.run_in_executor(None, lambda: index.search(q, k))

        retrieved: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=round(float(score), 4),
                )
            )

        return retrieved

    async def count(self) -> int:
        return len(self._chunks)

    async def reset(self) -> None:
        self._index = None
        self._chunks = []
        if self._index_path.exists():
            self._index_path.unlink()
        logger.warning("FAISS index reset")


# ── Factory ───────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """
    Return cached VectorStore singleton based on config.
    Use this everywhere — never instantiate directly.
    """
    settings = get_settings()
    if settings.vector_store_type == VectorStoreType.FAISS:
        logger.info("Using FAISS vector store")
        return FAISSVectorStore()
    logger.info("Using ChromaDB vector store")
    return ChromaVectorStore()
