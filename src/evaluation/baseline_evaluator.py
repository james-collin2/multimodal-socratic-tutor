"""
evaluation/baseline_evaluator.py

BaselineEvaluator — compares socratOT vs baselines for ACL paper.

Baselines:
  1. No-RAG:      plain LLM answer with no retrieved context
  2. No-Socratic: RAG answer but no hint/masking logic (direct answer)

Metrics: ROUGE-L, BERTScore F1, keyword overlap
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import logger


@dataclass
class BaselineResult:
    """Comparison results for one system."""

    system_name: str
    rouge_l: float = 0.0
    bert_score_f1: float = 0.0
    keyword_overlap: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "system": self.system_name,
            "rouge_l": round(self.rouge_l, 3),
            "bert_score_f1": round(self.bert_score_f1, 3),
            "keyword_overlap": round(self.keyword_overlap, 3),
            "n_samples": self.n_samples,
        }


class BaselineEvaluator:
    """
    Runs benchmark comparison across three systems.

    Systems compared:
      1. socratOT   — full RAG + Socratic logic
      2. no_rag     — LLM only, no retrieval
      3. no_socratic — RAG only, no hint masking
    """

    def __init__(self, n_samples: int = 10) -> None:
        self._n_samples = n_samples
        self._dataset = ROOT / "evaluation" / "ground_truth.jsonl"

    def _load_samples(self) -> list[dict]:
        samples = []
        with open(self._dataset) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))
        # Only anatomy questions: clinical_application (OT) questions ask for
        # knowledge outside the corpus, so answer-overlap metrics don't apply.
        samples = [s for s in samples if s.get("category", "anatomy") == "anatomy"]
        return samples[: self._n_samples]

    def _keyword_overlap(self, pred: str, ref: str) -> float:
        stops = {"the", "a", "an", "is", "are", "to", "of", "in", "and", "or"}

        def kws(t: str) -> set:
            return {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", t.lower()) if w not in stops}

        p, r = kws(pred), kws(ref)
        return len(p & r) / max(len(r), 1)

    def _rouge_l(self, pred: str, ref: str) -> float:
        """Simple ROUGE-L using LCS length."""

        def lcs(a: list, b: list) -> int:
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i - 1] == b[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        p_tokens = pred.lower().split()
        r_tokens = ref.lower().split()
        if not p_tokens or not r_tokens:
            return 0.0
        lcs_len = lcs(p_tokens, r_tokens)
        precision = lcs_len / len(p_tokens)
        recall = lcs_len / len(r_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _bert_score_approx(self, pred: str, ref: str) -> float:
        """Approximation using token overlap (actual BERTScore needs GPU)."""
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        if not pred_tokens or not ref_tokens:
            return 0.0
        precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
        recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    async def _get_socratot_answer(self, question: str) -> str:
        """Get answer from full socratOT system."""
        try:
            from src.core.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline()
            result = await pipeline.query(question)
            return result.answer
        except Exception as e:
            logger.warning("socratOT query failed: {e}", e=str(e))
            return ""

    async def _get_no_rag_answer(self, question: str) -> str:
        """Get answer from LLM with no retrieval context."""
        try:
            from src.models.llm_factory import get_llm

            llm = get_llm()
            loop = asyncio.get_event_loop()
            prompt = f"Answer this anatomy question directly: {question}"
            resp = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            logger.warning("No-RAG query failed: {e}", e=str(e))
            return ""

    async def _get_no_socratic_answer(self, question: str) -> str:
        """Get RAG answer with no Socratic masking (direct answer)."""
        try:
            from src.core.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline(
                system_prompt=(
                    "You are a helpful anatomy tutor. Answer questions directly and completely."
                )
            )
            result = await pipeline.query(question)
            return result.answer
        except Exception as e:
            logger.warning("No-Socratic query failed: {e}", e=str(e))
            return ""

    async def run(self) -> list[BaselineResult]:
        """Run all three systems and return comparison results."""
        samples = self._load_samples()
        print(f"  Running benchmark on {len(samples)} samples...")

        systems = {
            "socratOT": self._get_socratot_answer,
            "no_rag": self._get_no_rag_answer,
            "no_socratic": self._get_no_socratic_answer,
        }

        results: dict[str, dict] = {
            name: {"rouge": [], "bert": [], "overlap": []} for name in systems
        }

        for i, sample in enumerate(samples):
            q = sample["question"]
            ref = sample["reference_answer"]
            print(f"  Sample {i + 1}/{len(samples)}: {q[:50]}...", end="\r")

            for name, fn in systems.items():
                try:
                    answer = await fn(q)
                    if answer:
                        results[name]["rouge"].append(self._rouge_l(answer, ref))
                        results[name]["bert"].append(self._bert_score_approx(answer, ref))
                        results[name]["overlap"].append(self._keyword_overlap(answer, ref))
                except Exception as e:
                    logger.warning("System {n} failed: {e}", n=name, e=str(e))

        print()

        final = []
        for name, scores in results.items():
            n = len(scores["rouge"])
            final.append(
                BaselineResult(
                    system_name=name,
                    rouge_l=sum(scores["rouge"]) / max(n, 1),
                    bert_score_f1=sum(scores["bert"]) / max(n, 1),
                    keyword_overlap=sum(scores["overlap"]) / max(n, 1),
                    n_samples=n,
                )
            )

        return final


async def main() -> None:
    print("\n" + "=" * 60)
    print("  socratOT — Benchmark Comparison")
    print("=" * 60 + "\n")

    evaluator = BaselineEvaluator(n_samples=10)
    results = await evaluator.run()

    print(f"\n  {'System':<15} {'ROUGE-L':>8} {'BERTScore':>10} {'Overlap':>8}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 10} {'-' * 8}")
    for r in results:
        print(
            f"  {r.system_name:<15} "
            f"{r.rouge_l:>8.3f} "
            f"{r.bert_score_f1:>10.3f} "
            f"{r.keyword_overlap:>8.3f}"
        )

    out = ROOT / "evaluation" / "results" / "baseline_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    print(f"\n  Saved: {out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
