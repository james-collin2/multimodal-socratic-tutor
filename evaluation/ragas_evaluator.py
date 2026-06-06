"""
evaluation/ragas_evaluator.py

RagasEvaluator — runs RAGAS metrics on the ground truth dataset.

Metrics computed:
  faithfulness      — does answer stay grounded in retrieved context?
  answer_relevance  — does answer address the question?
  context_recall    — did retrieval find the right chunks?

Run: python evaluation/ragas_evaluator.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import logger


@dataclass
class RagasResult:
    """Results from one RAGAS evaluation run."""

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    n_samples: int = 0
    failed_samples: int = 0
    per_sample: list = field(default_factory=list)

    @property
    def average(self) -> float:
        return round((self.faithfulness + self.answer_relevance + self.context_recall) / 3, 3)

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "context_recall": round(self.context_recall, 3),
            "average": self.average,
            "n_samples": self.n_samples,
            "failed_samples": self.failed_samples,
        }


class RagasEvaluator:
    """
    Runs RAGAS evaluation on socratOT ground truth dataset.

    Args:
        dataset_path: Path to ground_truth.jsonl
        n_samples:    Number of QA pairs to evaluate (None = all)
    """

    def __init__(
        self,
        dataset_path: Path | None = None,
        n_samples: int | None = None,
    ) -> None:
        self._dataset_path = dataset_path or (ROOT / "evaluation" / "ground_truth.jsonl")
        self._n_samples = n_samples

    def _load_dataset(self) -> list[dict]:
        """
        Load ground truth QA pairs sampled evenly across topics.
        Excludes clinical_application — not in the anatomy corpus.
        """
        all_samples = []
        with open(self._dataset_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    if s.get("category") != "clinical_application":
                        all_samples.append(s)

        excluded = sum(
            1
            for line in open(self._dataset_path)
            if line.strip() and json.loads(line.strip()).get("category") == "clinical_application"
        )
        if excluded:
            logger.info(
                "Excluded {n} clinical-application questions (not in corpus)",
                n=excluded,
            )

        if not self._n_samples:
            logger.info("Loaded {n} evaluation samples", n=len(all_samples))
            return all_samples

        # Sample evenly across topics
        from collections import defaultdict

        by_topic: dict = defaultdict(list)
        for s in all_samples:
            by_topic[s.get("topic", "other")].append(s)

        topics = list(by_topic.keys())
        per_topic = max(1, self._n_samples // len(topics))
        sampled: list = []
        for topic in topics:
            sampled.extend(by_topic[topic][:per_topic])

        # Top up if needed
        remaining = [s for s in all_samples if s not in sampled]
        sampled.extend(remaining[: self._n_samples - len(sampled)])
        sampled = sampled[: self._n_samples]

        logger.info(
            "Loaded {n} samples across {t} topics",
            n=len(sampled),
            t=len(topics),
        )
        return sampled

    async def _get_rag_response(self, question: str) -> tuple[str, str, list[str]]:
        """
        Run question through RAG pipeline.
        Returns (answer, context, citations).
        """
        try:
            from src.core.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline()
            result = await pipeline.query(question)
            return (
                result.answer,
                result.retrieval.assembled_context,
                result.citations,
            )
        except Exception as e:
            logger.warning("RAG failed for question: {e}", e=str(e))
            return "", "", []

    async def run(self) -> RagasResult:
        """Run full RAGAS evaluation. Returns RagasResult."""
        samples = self._load_dataset()

        try:
            return await self._run_ragas_library(samples)
        except ImportError:
            logger.warning("ragas library unavailable — using manual metrics")
            return await self._run_manual_metrics(samples)
        except Exception as e:
            logger.warning("RAGAS failed: {e} — using manual metrics", e=str(e))
            return await self._run_manual_metrics(samples)

    async def _run_ragas_library(self, samples: list[dict]) -> RagasResult:
        """Use official ragas library for evaluation."""
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness

        print(f"  Running RAGAS on {len(samples)} samples...")

        rows = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for i, s in enumerate(samples):
            print(f"  Getting RAG response {i + 1}/{len(samples)}...", end="\r")
            answer, context, _ = await self._get_rag_response(s["question"])
            rows["question"].append(s["question"])
            rows["answer"].append(answer or "No answer retrieved.")
            rows["contexts"].append([context] if context else ["No context retrieved."])
            rows["ground_truth"].append(s["reference_answer"])

        print()
        dataset = Dataset.from_dict(rows)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_recall],
            ),
        )

        df = result.to_pandas()
        return RagasResult(
            faithfulness=float(df["faithfulness"].mean()),
            answer_relevance=float(df["answer_relevancy"].mean()),
            context_recall=float(df["context_recall"].mean()),
            n_samples=len(samples),
            failed_samples=int(df.isnull().any(axis=1).sum()),
            per_sample=df.to_dict(orient="records"),
        )

    async def _run_manual_metrics(self, samples: list[dict]) -> RagasResult:
        """
        Manual metric computation when ragas library unavailable.
        Uses keyword overlap as proxy for faithfulness and relevance.
        """
        import re

        def keywords(text: str) -> set:
            stops = {"the", "a", "an", "is", "are", "to", "of", "in", "and", "or", "it", "its"}
            return {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()) if w not in stops}

        faith_scores, rel_scores, recall_scores = [], [], []
        per_sample = []
        failed = 0

        for i, s in enumerate(samples):
            print(f"  Evaluating sample {i + 1}/{len(samples)}...", end="\r")
            try:
                answer, context, _ = await self._get_rag_response(s["question"])

                if not answer or not context:
                    failed += 1
                    continue

                ans_kw = keywords(answer)
                ctx_kw = keywords(context)
                ref_kw = keywords(s["reference_answer"])
                q_kw = keywords(s["question"])

                # Faithfulness: how much of the answer is in the context
                faith = len(ans_kw & ctx_kw) / max(len(ans_kw), 1)

                # Answer relevance: how much answer addresses the question
                relevance = len(ans_kw & q_kw) / max(len(q_kw), 1)

                # Context recall: how much of reference answer is in context
                recall = len(ref_kw & ctx_kw) / max(len(ref_kw), 1)

                faith_scores.append(faith)
                rel_scores.append(relevance)
                recall_scores.append(recall)

                per_sample.append(
                    {
                        "question": s["question"][:60],
                        "faithfulness": round(faith, 3),
                        "answer_relevancy": round(relevance, 3),
                        "context_recall": round(recall, 3),
                    }
                )
            except Exception as e:
                logger.warning("Sample {i} failed: {e}", i=i, e=str(e))
                failed += 1

        print()
        n = len(faith_scores)
        return RagasResult(
            faithfulness=sum(faith_scores) / max(n, 1),
            answer_relevance=sum(rel_scores) / max(n, 1),
            context_recall=sum(recall_scores) / max(n, 1),
            n_samples=len(samples),
            failed_samples=failed,
            per_sample=per_sample,
        )


async def main() -> None:
    print("\n" + "=" * 60)
    print("  socratOT — RAGAS Evaluation")
    print("=" * 60 + "\n")

    evaluator = RagasEvaluator(n_samples=20)
    result = await evaluator.run()

    print("\n" + "=" * 60)
    print("  RAGAS Results")
    print("=" * 60)
    print(f"  Faithfulness:     {result.faithfulness:.3f}")
    print(f"  Answer relevance: {result.answer_relevance:.3f}")
    print(f"  Context recall:   {result.context_recall:.3f}")
    print(f"  Average:          {result.average:.3f}")
    print(f"  Samples:          {result.n_samples} ({result.failed_samples} failed)")
    print("=" * 60 + "\n")

    # Save results
    out = ROOT / "evaluation" / "results" / "ragas_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"  Saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
