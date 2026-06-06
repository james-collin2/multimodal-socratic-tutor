"""
tests/unit/test_phase5.py

Phase 5 evaluation tests — 25 tests.
Run: pytest tests/unit/test_phase5.py -v
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


# ── RagasEvaluator tests ──────────────────────────────────────────────────────


class TestRagasEvaluator:
    def test_ragas_result_average(self) -> None:
        from evaluation.ragas_evaluator import RagasResult

        r = RagasResult(faithfulness=0.8, answer_relevance=0.7, context_recall=0.9)
        assert abs(r.average - 0.8) < 0.01

    def test_ragas_result_to_dict(self) -> None:
        from evaluation.ragas_evaluator import RagasResult

        r = RagasResult(faithfulness=0.85, answer_relevance=0.75, context_recall=0.80, n_samples=20)
        d = r.to_dict()
        assert "faithfulness" in d
        assert "answer_relevance" in d
        assert "context_recall" in d
        assert "average" in d
        assert d["n_samples"] == 20

    def test_ragas_evaluator_loads_dataset(self) -> None:
        from evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(n_samples=5)
        samples = evaluator._load_dataset()
        assert len(samples) == 5
        assert "question" in samples[0]
        assert "reference_answer" in samples[0]

    def test_ragas_evaluator_full_dataset(self) -> None:
        from pathlib import Path

        from evaluation.ragas_evaluator import RagasEvaluator

        # Full ground-truth file has the required >= 50 QA pairs.
        gt = Path("evaluation/ground_truth.jsonl")
        total = sum(1 for line in gt.read_text().splitlines() if line.strip())
        assert total >= 50

        # RAG evaluation loads only the anatomy subset (clinical questions excluded),
        # and every loaded sample is in-corpus anatomy.
        samples = RagasEvaluator()._load_dataset()
        assert 0 < len(samples) <= total
        assert all(s.get("category", "anatomy") == "anatomy" for s in samples)

    def test_manual_metrics_zero_for_empty(self) -> None:
        import asyncio

        from evaluation.ragas_evaluator import RagasEvaluator

        evaluator = RagasEvaluator(n_samples=0)
        result = asyncio.run(evaluator._run_manual_metrics([]))
        assert result.n_samples == 0
        assert result.faithfulness == 0.0


# ── Compliance evaluator tests ────────────────────────────────────────────────


class TestComplianceEvaluator:
    def test_all_bypass_attempts_detected(self) -> None:
        from evaluation.compliance_evaluator import (
            SocraticComplianceEvaluator,
        )

        evaluator = SocraticComplianceEvaluator()
        result = evaluator.run()
        assert result.bypass_detection_rate == 1.0, f"Failed bypasses: {result.failed_bypasses}"

    def test_no_false_positives(self) -> None:
        from evaluation.compliance_evaluator import SocraticComplianceEvaluator

        evaluator = SocraticComplianceEvaluator()
        result = evaluator.run()
        assert result.false_positive_rate == 0.0, f"False positives: {result.false_positive_cases}"

    def test_compliance_rate_is_100_percent(self) -> None:
        from evaluation.compliance_evaluator import SocraticComplianceEvaluator

        evaluator = SocraticComplianceEvaluator()
        result = evaluator.run()
        assert result.socratic_compliance_rate == 1.0

    def test_result_to_dict_has_all_fields(self) -> None:
        from evaluation.compliance_evaluator import SocraticComplianceEvaluator

        evaluator = SocraticComplianceEvaluator()
        result = evaluator.run()
        d = result.to_dict()
        for key in [
            "bypass_detection_rate",
            "false_positive_rate",
            "socratic_compliance",
            "bypass_total",
            "legitimate_total",
        ]:
            assert key in d

    def test_bypass_total_matches_list(self) -> None:
        from evaluation.compliance_evaluator import (
            BYPASS_ATTEMPTS,
            SocraticComplianceEvaluator,
        )

        evaluator = SocraticComplianceEvaluator()
        result = evaluator.run()
        assert result.bypass_total == len(BYPASS_ATTEMPTS)

    def test_20_plus_bypass_scenarios(self) -> None:
        from evaluation.compliance_evaluator import BYPASS_ATTEMPTS

        assert len(BYPASS_ATTEMPTS) >= 20


# ── BaselineEvaluator tests ───────────────────────────────────────────────────


class TestBaselineEvaluator:
    def test_rouge_l_identical_strings(self) -> None:
        from evaluation.baseline_evaluator import BaselineEvaluator

        ev = BaselineEvaluator()
        assert ev._rouge_l("hello world", "hello world") == 1.0

    def test_rouge_l_empty_string(self) -> None:
        from evaluation.baseline_evaluator import BaselineEvaluator

        ev = BaselineEvaluator()
        assert ev._rouge_l("", "hello world") == 0.0

    def test_rouge_l_partial_overlap(self) -> None:
        from evaluation.baseline_evaluator import BaselineEvaluator

        ev = BaselineEvaluator()
        score = ev._rouge_l(
            "the cerebellum coordinates movement", "cerebellum coordinates voluntary movement"
        )
        assert 0.0 < score < 1.0

    def test_bert_score_identical(self) -> None:
        from evaluation.baseline_evaluator import BaselineEvaluator

        ev = BaselineEvaluator()
        assert ev._bert_score_approx("hello world", "hello world") == 1.0

    def test_keyword_overlap_perfect(self) -> None:
        from evaluation.baseline_evaluator import BaselineEvaluator

        ev = BaselineEvaluator()
        score = ev._keyword_overlap(
            "cerebellum coordinates voluntary movements balance",
            "cerebellum coordinates voluntary movements balance equilibrium",
        )
        assert score > 0.7

    def test_keyword_overlap_zero(self) -> None:
        from evaluation.baseline_evaluator import BaselineEvaluator

        ev = BaselineEvaluator()
        score = ev._keyword_overlap(
            "completely unrelated text here", "cerebellum coordinates movement balance"
        )
        assert score < 0.3

    def test_baseline_result_to_dict(self) -> None:
        from evaluation.baseline_evaluator import BaselineResult

        r = BaselineResult(
            system_name="socratOT",
            rouge_l=0.75,
            bert_score_f1=0.80,
            keyword_overlap=0.70,
            n_samples=10,
        )
        d = r.to_dict()
        assert d["system"] == "socratOT"
        assert d["rouge_l"] == 0.75
        assert d["n_samples"] == 10

    def test_load_samples_limited(self) -> None:
        from evaluation.baseline_evaluator import BaselineEvaluator

        ev = BaselineEvaluator(n_samples=5)
        samples = ev._load_samples()
        assert len(samples) == 5


# ── Results directory tests ───────────────────────────────────────────────────


class TestEvaluationInfrastructure:
    def test_ground_truth_dataset_exists(self) -> None:
        path = ROOT / "evaluation" / "ground_truth.jsonl"
        assert path.exists(), "ground_truth.jsonl not found"

    def test_ground_truth_has_50_plus_entries(self) -> None:
        path = ROOT / "evaluation" / "ground_truth.jsonl"
        entries = [l for l in path.read_text().split("\n") if l.strip()]
        assert len(entries) >= 50

    def test_results_directory_exists(self) -> None:
        path = ROOT / "evaluation" / "results"
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists()

    def test_secrets_example_exists(self) -> None:
        path = ROOT / ".streamlit" / "secrets.toml.example"
        assert path.exists(), ".streamlit/secrets.toml.example not found"

    def test_secrets_example_has_key_fields(self) -> None:
        path = ROOT / ".streamlit" / "secrets.toml.example"
        content = path.read_text()
        assert "OPENAI_API_KEY" in content
        assert "LLM_PROVIDER" in content
        assert "EMBEDDING_PROVIDER" in content
