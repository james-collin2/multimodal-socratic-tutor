"""
src/models/embedding_model.py

Multi-provider embedding model wrapper.
Supports:
  openai              → text-embedding-3-small (recommended)
  sentence-transformers → all-MiniLM-L6-v2 (local fallback)

All encode() calls are async — run in thread pool to avoid blocking.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from src.config.settings import EmbeddingProvider, get_settings
from src.utils.exceptions import EmbeddingError
from src.utils.logger import logger


class EmbeddingModel:
    """
    Async embedding wrapper supporting OpenAI and SentenceTransformers.
    Loaded lazily on first encode() call.
    """

    def __init__(self) -> None:
        self._model: object = None
        self._client: object = None
        self._provider: str = ""
        self._model_name: str = ""
        self._dim: int = 0

    def _load(self) -> None:
        """Lazily initialise the embedding backend."""
        if self._model is not None or self._client is not None:
            return

        settings = get_settings()
        self._provider = settings.embedding_provider.value

        if settings.embedding_provider == EmbeddingProvider.OPENAI:
            self._load_openai(settings)
        else:
            self._load_sentence_transformers(settings)

    def _load_openai(self, settings: object) -> None:
        """Load OpenAI embedding client."""
        try:
            from openai import OpenAI

            if not settings.openai_api_key:
                raise EmbeddingError(
                    "OPENAI_API_KEY not set",
                    detail=(
                        "Set OPENAI_API_KEY in .env or switch "
                        "EMBEDDING_PROVIDER=sentence-transformers"
                    ),
                )
            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model_name = settings.openai_embedding_model
            self._dim = 1536  # text-embedding-3-small default
            logger.info("Embedding: OpenAI {m}", m=self._model_name)
        except ImportError as e:
            raise EmbeddingError(
                "openai package not installed",
                detail="Run: pip install openai",
            ) from e

    def _load_sentence_transformers(self, settings: object) -> None:
        """Load SentenceTransformer model (local fallback)."""
        try:
            from sentence_transformers import SentenceTransformer

            device = self._resolve_device(settings.embedding_device)
            self._model_name = settings.embedding_model
            self._model = SentenceTransformer(self._model_name, device=device)
            self._dim = 384
            logger.info(
                "Embedding: SentenceTransformer {m} on {d}",
                m=self._model_name,
                d=device,
            )
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers not installed",
                detail=str(e),
            ) from e

    @staticmethod
    def _resolve_device(device: object) -> str:
        """Resolve embedding device with fallback to CPU."""
        device_val = device.value if hasattr(device, "value") else str(device)
        if device_val == "mps":
            try:
                import torch

                return "mps" if torch.backends.mps.is_available() else "cpu"
            except ImportError:
                return "cpu"
        if device_val == "cuda":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return "cpu"

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Async encode a list of texts into embedding vectors.
        Offloads CPU-bound work to thread pool.
        """
        self._load()
        loop = asyncio.get_running_loop()

        try:
            if self._provider == EmbeddingProvider.OPENAI.value:
                return await loop.run_in_executor(
                    None,
                    lambda: self._encode_openai(texts),
                )
            return await loop.run_in_executor(
                None,
                lambda: self._encode_st(texts),
            )
        except Exception as e:
            raise EmbeddingError("Embedding failed", detail=str(e)) from e

    def _encode_openai(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI Embeddings API."""
        response = self._client.embeddings.create(  # type: ignore[union-attr]
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _encode_st(self, texts: list[str]) -> list[list[float]]:
        """Encode with SentenceTransformer."""
        embeddings = self._model.encode(  # type: ignore[union-attr]
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    async def encode_single(self, text: str) -> list[float]:
        """Encode a single string."""
        results = await self.encode([text])
        return results[0]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        """Embedding dimension."""
        return self._dim


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Cached EmbeddingModel singleton."""
    return EmbeddingModel()
