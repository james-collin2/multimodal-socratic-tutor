"""
tests/unit/core/test_phase2.py

Phase 2 milestone verification tests.
All tests must pass before moving to Phase 3.

Run: pytest tests/unit/core/test_phase2.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ── Chunker tests ─────────────────────────────────────────────────────────────


class TestChunker:
    def test_chunk_basic_text(self) -> None:
        from src.core.rag.chunker import Chunker

        chunker = Chunker(chunk_size=128, chunk_overlap=16)
        text = "The cerebellum coordinates voluntary movements. " * 50
        chunks = chunker.chunk_text(text, source="test/doc.txt")
        assert len(chunks) > 0

    def test_chunk_preserves_source(self) -> None:
        from src.core.rag.chunker import Chunker

        chunker = Chunker()
        text = "Anatomy text about the brain and cerebellum structure. " * 20
        chunks = chunker.chunk_text(text, source="openStax/chapter1.txt")
        assert all(c.source == "openStax/chapter1.txt" for c in chunks)

    def test_chunk_with_topic_tags(self) -> None:
        from src.core.rag.chunker import Chunker

        chunker = Chunker()
        text = "The cranial nerves emerge from the brainstem. " * 20
        chunks = chunker.chunk_text(text, source="test.txt", topic_tags=["cranial_nerves"])
        assert all("cranial_nerves" in c.topic_tags for c in chunks)

    def test_chunk_empty_text_returns_empty(self) -> None:
        from src.core.rag.chunker import Chunker

        chunker = Chunker()
        chunks = chunker.chunk_text("", source="empty.txt")
        assert chunks == []

    def test_chunk_tiny_text_returns_empty(self) -> None:
        from src.core.rag.chunker import Chunker

        chunker = Chunker()
        chunks = chunker.chunk_text("hi", source="tiny.txt")
        assert chunks == []

    def test_chunk_has_token_count(self) -> None:
        from src.core.rag.chunker import Chunker

        chunker = Chunker()
        text = "The spinal cord carries sensory and motor information. " * 30
        chunks = chunker.chunk_text(text, source="test.txt")
        assert all(c.token_count > 0 for c in chunks)

    def test_chunk_has_metadata(self) -> None:
        from src.core.rag.chunker import Chunker

        chunker = Chunker()
        text = "Purkinje cells are the primary output neurons. " * 20
        chunks = chunker.chunk_text(text, source="test.txt")
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_properties(self) -> None:
        from src.core.rag.chunker import Chunker

        chunker = Chunker(chunk_size=256, chunk_overlap=32)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 32


# ── Embedder tests ────────────────────────────────────────────────────────────


class TestEmbedder:
    def test_embedder_has_correct_batch_size(self) -> None:
        from src.core.rag.embedder import Embedder

        embedder = Embedder(batch_size=16)
        assert embedder.batch_size == 16

    def test_embed_empty_chunks_returns_empty(self) -> None:
        import asyncio

        from src.core.rag.embedder import Embedder

        embedder = Embedder()
        result = asyncio.run(embedder.embed_chunks([]))
        assert result == []


# ── HallucinationGuard tests ──────────────────────────────────────────────────


class TestHallucinationGuard:
    def test_grounded_answer_passes(self) -> None:
        import asyncio

        from src.core.rag.hallucination_guard import HallucinationGuard

        guard = HallucinationGuard(llm=None, use_llm_verification=False)
        context = """The cerebellum coordinates voluntary movements and maintains
        balance. Purkinje cells are the primary output neurons of the cerebellar
        cortex. The cerebellum contains the dentate nucleus."""
        answer = "The cerebellum coordinates voluntary movements and contains Purkinje cells."
        result = asyncio.run(guard.check(answer=answer, context=context))
        assert result.is_grounded is True
        assert result.confidence > 0.0

    def test_hallucinated_answer_flagged(self) -> None:
        import asyncio

        from src.core.rag.hallucination_guard import HallucinationGuard

        guard = HallucinationGuard(
            llm=None,
            use_llm_verification=False,
            overlap_threshold=0.8,
        )
        context = "The cerebellum is located at the posterior brain."
        answer = "The hippocampus secretes testosterone to regulate metabolism."
        result = asyncio.run(guard.check(answer=answer, context=context))
        assert result.confidence < 0.5

    def test_empty_context_fails(self) -> None:
        import asyncio

        from src.core.rag.hallucination_guard import HallucinationGuard

        guard = HallucinationGuard(llm=None, use_llm_verification=False)
        result = asyncio.run(guard.check(answer="Some answer", context=""))
        assert result.is_grounded is False

    def test_keyword_overlap_calculation(self) -> None:
        from src.core.rag.hallucination_guard import HallucinationGuard

        guard = HallucinationGuard(llm=None, use_llm_verification=False)
        score = guard._keyword_overlap(
            "cerebellum coordinates voluntary movements balance",
            "cerebellum coordinates voluntary movements balance equilibrium",
        )
        assert score > 0.7


# ── RAG Schema tests ──────────────────────────────────────────────────────────


class TestRAGSchemas:
    def test_document_chunk_creation(self) -> None:
        from src.schemas.rag import DocumentChunk

        chunk = DocumentChunk(
            content="The cerebellum coordinates movement.",
            source="openStax/ch14.txt",
            topic_tags=["cerebellum"],
            token_count=8,
        )
        assert chunk.content == "The cerebellum coordinates movement."
        assert chunk.source == "openStax/ch14.txt"
        assert "cerebellum" in chunk.topic_tags

    def test_retrieval_result_empty(self) -> None:
        from src.schemas.rag import RetrievalResult

        result = RetrievalResult(query="test query")
        assert result.has_results is False
        assert result.top_score == 0.0
        assert result.assembled_context == ""

    def test_retrieval_result_with_chunks(self) -> None:
        from src.schemas.rag import DocumentChunk, RetrievalResult, RetrievedChunk

        chunk = DocumentChunk(content="test", source="test.txt")
        retrieved = RetrievedChunk(chunk=chunk, score=0.85)
        result = RetrievalResult(
            query="test",
            chunks=[retrieved],
            assembled_context="test context",
            citations=["test.txt"],
        )
        assert result.has_results is True
        assert result.top_score == 0.85

    def test_retrieved_chunk_score_range(self) -> None:
        import pydantic

        from src.schemas.rag import DocumentChunk, RetrievedChunk

        chunk = DocumentChunk(content="test", source="test.txt")
        with pytest.raises((pydantic.ValidationError, ValueError)):
            RetrievedChunk(chunk=chunk, score=1.5)


# ── Ground truth dataset tests ────────────────────────────────────────────────


class TestGroundTruthDataset:
    DATASET_PATH = ROOT / "evaluation" / "ground_truth.jsonl"

    def test_dataset_exists(self) -> None:
        assert self.DATASET_PATH.exists(), "evaluation/ground_truth.jsonl not found"

    def test_dataset_has_50_plus_entries(self) -> None:
        entries = self.DATASET_PATH.read_text().strip().split("\n")
        assert len(entries) >= 50, f"Need ≥50 QA pairs, found {len(entries)}"

    def test_all_entries_valid_json(self) -> None:
        for i, line in enumerate(self.DATASET_PATH.read_text().strip().split("\n")):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {i + 1} is not valid JSON: {e}")

    def test_all_entries_have_required_fields(self) -> None:
        required = {"question", "reference_answer", "source", "topic", "difficulty"}
        for i, line in enumerate(self.DATASET_PATH.read_text().strip().split("\n")):
            entry = json.loads(line)
            missing = required - set(entry.keys())
            assert not missing, f"Line {i + 1} missing fields: {missing}"

    def test_difficulty_values_valid(self) -> None:
        valid = {"beginner", "intermediate", "advanced"}
        for line in self.DATASET_PATH.read_text().strip().split("\n"):
            entry = json.loads(line)
            assert entry["difficulty"] in valid, f"Invalid difficulty: {entry['difficulty']}"

    def test_topics_match_taxonomy(self) -> None:
        import yaml

        topics_path = ROOT / "config" / "topics.yaml"
        with open(topics_path) as f:
            taxonomy = yaml.safe_load(f)

        all_topics = set()
        for category in taxonomy.values():
            for item in category:
                all_topics.add(item["name"])

        for line in self.DATASET_PATH.read_text().strip().split("\n"):
            entry = json.loads(line)
            assert entry["topic"] in all_topics, f"Unknown topic: {entry['topic']}"

    def test_covers_multiple_topics(self) -> None:
        topics = set()
        for line in self.DATASET_PATH.read_text().strip().split("\n"):
            topics.add(json.loads(line)["topic"])
        assert len(topics) >= 4, f"Need ≥4 topics covered, found {len(topics)}: {topics}"

    def test_questions_are_non_empty(self) -> None:
        for line in self.DATASET_PATH.read_text().strip().split("\n"):
            entry = json.loads(line)
            assert len(entry["question"]) > 10, f"Question too short: {entry['question']}"
            assert len(entry["reference_answer"]) > 20, f"Answer too short for: {entry['question']}"


# ── Ingest corpus sample test ─────────────────────────────────────────────────


class TestCorpusIngestion:
    def test_create_sample_corpus(self, tmp_path: Path) -> None:
        from scripts.ingest_corpus import create_sample_corpus

        files = create_sample_corpus(tmp_path)
        assert len(files) >= 5
        for f in files:
            assert f.exists()
            assert f.stat().st_size > 100

    def test_create_image_metadata(self, tmp_path: Path) -> None:
        from scripts.ingest_corpus import create_image_metadata

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        meta_path = create_image_metadata(images_dir)
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert len(data) >= 5
        for item in data:
            assert "filename" in item
            assert "structures" in item
            assert "license" in item
