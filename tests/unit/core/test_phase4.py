"""
tests/unit/core/test_phase4.py

Phase 4 milestone verification — 30+ tests.
Run: pytest tests/unit/core/test_phase4.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ── VisionAnalyzer tests ──────────────────────────────────────────────────────


class TestVisionAnalyzer:
    def test_parse_valid_json_response(self) -> None:
        from src.core.multimodal.vision_analyzer import VisionAnalyzer

        analyzer = VisionAnalyzer(llm=None)
        raw = json.dumps(
            {
                "structures": ["cerebellum", "vermis", "folia"],
                "region": "brain",
                "confidence": 0.85,
                "description": "Diagram of cerebellar anatomy.",
                "ot_relevance": "Motor coordination disorders.",
            }
        )
        result = analyzer._parse_result(raw)
        assert result.structures == ["cerebellum", "vermis", "folia"]
        assert result.region == "brain"
        assert result.confidence == 0.85

    def test_parse_invalid_json_uses_fallback(self) -> None:
        from src.core.multimodal.vision_analyzer import VisionAnalyzer

        analyzer = VisionAnalyzer(llm=None)
        result = analyzer._parse_result("The image shows the cerebellum and median nerve.")
        assert isinstance(result.structures, list)
        assert result.confidence == 0.4

    def test_extract_structures_from_text(self) -> None:
        from src.core.multimodal.vision_analyzer import VisionAnalyzer

        structures = VisionAnalyzer._extract_structures_from_text(
            "This image shows the cerebellum, brachial plexus, and median nerve."
        )
        assert "cerebellum" in structures
        assert "median nerve" in structures

    def test_vision_result_error_flag(self) -> None:
        from src.core.multimodal.vision_analyzer import VisionResult

        result = VisionResult(
            structures=[],
            region="unknown",
            confidence=0.0,
            description="",
            ot_relevance="",
            raw_response="",
            error="File not found",
        )
        assert result.error is not None

    def test_vision_result_no_error(self) -> None:
        from src.core.multimodal.vision_analyzer import VisionResult

        result = VisionResult(
            structures=["cerebellum"],
            region="brain",
            confidence=0.9,
            description="Test",
            ot_relevance="OT",
            raw_response="{}",
            error=None,
        )
        assert result.error is None
        assert len(result.structures) == 1


# ── ImageQuestionGenerator tests ─────────────────────────────────────────────


class TestImageQuestionGenerator:
    def test_fallback_questions_returned_for_brain(self) -> None:
        from src.core.multimodal.question_generator import ImageQuestionGenerator

        gen = ImageQuestionGenerator(llm=None)
        qs = gen._fallback_questions("brain")
        assert len(qs) >= 1
        assert all(q.question for q in qs)

    def test_fallback_questions_have_hints(self) -> None:
        from src.core.multimodal.question_generator import ImageQuestionGenerator

        gen = ImageQuestionGenerator(llm=None)
        qs = gen._fallback_questions("hand")
        assert all(q.hint for q in qs)

    def test_fallback_for_unknown_region(self) -> None:
        from src.core.multimodal.question_generator import ImageQuestionGenerator

        gen = ImageQuestionGenerator(llm=None)
        qs = gen._fallback_questions("unknown_region")
        assert len(qs) >= 1

    def test_question_dataclass_fields(self) -> None:
        from src.core.multimodal.question_generator import ImageQuestion

        q = ImageQuestion(
            question="What is the role of this structure?",
            structure="cerebellum",
            difficulty="intermediate",
            hint="Think about motor coordination.",
        )
        assert q.difficulty == "intermediate"

    @pytest.mark.asyncio
    async def test_generate_empty_structures_returns_fallback(self) -> None:
        from src.core.multimodal.question_generator import ImageQuestionGenerator
        from src.core.multimodal.vision_analyzer import VisionResult

        gen = ImageQuestionGenerator(llm=None)
        vision = VisionResult(
            structures=[],
            region="hand",
            confidence=0.5,
            description="",
            ot_relevance="",
            raw_response="",
        )
        qs = await gen.generate(vision)
        assert len(qs) >= 1


# ── StudentMemory tests ───────────────────────────────────────────────────────


class TestStudentMemory:
    """Each test creates its own StudentMemory with a unique DB path."""

    @pytest.mark.asyncio
    async def test_load_returns_empty_for_new_student(self, tmp_path) -> None:
        import os

        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/m1.db"
        from src.config.settings import get_settings

        get_settings.cache_clear()
        from src.core.memory.student_memory import StudentMemory

        memory = StudentMemory()
        record = await memory.load("new_student")
        assert record.student_id == "new_student"
        assert record.weak_topics == []
        assert record.mastery_scores == {}

    @pytest.mark.asyncio
    async def test_save_and_reload(self, tmp_path) -> None:
        import os

        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/m2.db"
        from src.config.settings import get_settings

        get_settings.cache_clear()
        from src.core.memory.student_memory import MemoryRecord, StudentMemory

        memory = StudentMemory()
        record = MemoryRecord(
            student_id="student_x",
            weak_topics=["cerebellum"],
            mastery_scores={"cerebellum": 45.0},
        )
        await memory.save(record)
        loaded = await memory.load("student_x")
        assert "cerebellum" in loaded.weak_topics
        assert loaded.mastery_scores["cerebellum"] == 45.0

    @pytest.mark.asyncio
    async def test_update_topic_score_updates_weak_list(self, tmp_path) -> None:
        import os

        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/m3.db"
        from src.config.settings import get_settings

        get_settings.cache_clear()
        from src.core.memory.student_memory import StudentMemory

        memory = StudentMemory()
        record = await memory.update_topic_score("student_y", "hand_anatomy", 40.0)
        assert "hand_anatomy" in record.weak_topics
        assert "hand_anatomy" not in record.strong_topics

    @pytest.mark.asyncio
    async def test_update_topic_score_updates_strong_list(self, tmp_path) -> None:
        import os

        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/m4.db"
        from src.config.settings import get_settings

        get_settings.cache_clear()
        from src.core.memory.student_memory import StudentMemory

        memory = StudentMemory()
        record = await memory.update_topic_score("student_z", "cerebellum", 90.0)
        assert "cerebellum" in record.strong_topics
        assert "cerebellum" not in record.weak_topics

    def test_memory_record_to_dict(self) -> None:
        from src.core.memory.student_memory import MemoryRecord

        record = MemoryRecord(
            student_id="test",
            weak_topics=["cn7"],
            mastery_scores={"cn7": 50.0},
        )
        d = record.to_dict()
        assert d["student_id"] == "test"
        assert json.loads(d["weak_topics"]) == ["cn7"]


# ── MemoryManager tests ───────────────────────────────────────────────────────


class TestMemoryManager:
    """Each test creates its own manager with a unique DB path."""

    @pytest.mark.asyncio
    async def test_new_student_gets_welcome_opener(self, tmp_path) -> None:
        import os

        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/mgr1.db"
        from src.config.settings import get_settings

        get_settings.cache_clear()
        from src.core.memory.memory_manager import MemoryManager
        from src.core.memory.student_memory import StudentMemory

        manager = MemoryManager(memory=StudentMemory())
        ctx = await manager.load_session_context("brand_new", "Alice")
        assert not ctx.is_returning
        assert "Alice" in ctx.personalised_opener or "Welcome" in ctx.personalised_opener

    @pytest.mark.asyncio
    async def test_returning_student_gets_personalised_opener(self, tmp_path) -> None:
        import os

        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/mgr2.db"
        from src.config.settings import get_settings

        get_settings.cache_clear()
        from src.core.memory.memory_manager import MemoryManager
        from src.core.memory.student_memory import MemoryRecord, StudentMemory

        memory = StudentMemory()
        manager = MemoryManager(memory=memory)
        record = MemoryRecord(
            student_id="returning_1",
            weak_topics=["cranial_nerves"],
            total_sessions=2,
        )
        await memory.save(record)
        ctx = await manager.load_session_context("returning_1", "Bob")
        assert ctx.is_returning
        assert (
            "cranial" in ctx.personalised_opener.lower()
            or "challenging" in ctx.personalised_opener.lower()
            or "back" in ctx.personalised_opener.lower()
        )

    @pytest.mark.asyncio
    async def test_priority_topics_ordered_by_score(self, tmp_path) -> None:
        import os

        os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tmp_path}/mgr3.db"
        from src.config.settings import get_settings

        get_settings.cache_clear()
        from src.core.memory.memory_manager import MemoryManager
        from src.core.memory.student_memory import MemoryRecord, StudentMemory

        memory = StudentMemory()
        manager = MemoryManager(memory=memory)
        record = MemoryRecord(
            student_id="priority_test",
            weak_topics=["spinal_cord", "hand_anatomy"],
            mastery_scores={"spinal_cord": 35.0, "hand_anatomy": 50.0},
            total_sessions=1,
        )
        await memory.save(record)
        ctx = await manager.load_session_context("priority_test")
        assert ctx.priority_topics[0] == "spinal_cord"


# ── ClinicalScenarioGenerator tests ──────────────────────────────────────────


class TestClinicalScenarioGenerator:
    def test_fallback_scenario_has_required_fields(self) -> None:
        from src.core.assessment.scenario_generator import ClinicalScenarioGenerator

        gen = ClinicalScenarioGenerator(llm=None)
        scenario = gen._fallback_scenario("cerebellum", "A patient...", "intermediate")
        assert scenario.topic == "cerebellum"
        assert len(scenario.question) > 10
        assert scenario.difficulty == "intermediate"

    def test_scenario_template_exists_for_key_topics(self) -> None:
        from src.core.assessment.scenario_generator import SCENARIO_TEMPLATES

        for topic in ["cerebellum", "cranial_nerves", "spinal_cord"]:
            assert topic in SCENARIO_TEMPLATES

    def test_clinical_scenario_schema(self) -> None:
        from src.schemas.assessment import ClinicalScenario

        s = ClinicalScenario(
            topic="cerebellum",
            scenario_text="Patient presents with ataxia.",
            question="Which structures are affected?",
            reference_answer="The cerebellar cortex and deep nuclei.",
        )
        assert s.topic == "cerebellum"


# ── ReasoningEvaluator tests ──────────────────────────────────────────────────


class TestReasoningEvaluator:
    @pytest.mark.asyncio
    async def test_empty_response_scores_zero(self) -> None:
        from src.core.assessment.reasoning_evaluator import ReasoningEvaluator
        from src.schemas.assessment import ClinicalScenario, MasteryLevel

        evaluator = ReasoningEvaluator(llm=None)
        scenario = ClinicalScenario(
            topic="cerebellum",
            scenario_text="Patient has ataxia.",
            question="What is affected?",
            reference_answer="The cerebellum coordinates movement.",
        )
        result = await evaluator.evaluate("", scenario)
        assert result.total == 0
        assert result.mastery_level == MasteryLevel.NOVICE

    @pytest.mark.asyncio
    async def test_good_response_scores_above_20(self) -> None:
        from src.core.assessment.reasoning_evaluator import ReasoningEvaluator
        from src.schemas.assessment import ClinicalScenario

        evaluator = ReasoningEvaluator(llm=None)
        scenario = ClinicalScenario(
            topic="cerebellum",
            scenario_text="Patient has intention tremor and ataxia.",
            question="Which structure is affected?",
            reference_answer="The cerebellum is affected causing ataxia and intention tremor.",
        )
        result = await evaluator.evaluate(
            "The cerebellum is affected causing ataxia and intention tremor.",
            scenario,
        )
        assert result.total > 20

    def test_score_to_level_mapping(self) -> None:
        from src.core.assessment.reasoning_evaluator import ReasoningEvaluator
        from src.schemas.assessment import MasteryLevel

        assert ReasoningEvaluator._score_to_level(90) == MasteryLevel.MASTERY
        assert ReasoningEvaluator._score_to_level(75) == MasteryLevel.PROFICIENT
        assert ReasoningEvaluator._score_to_level(55) == MasteryLevel.DEVELOPING
        assert ReasoningEvaluator._score_to_level(30) == MasteryLevel.NOVICE

    def test_reasoning_score_passed_property(self) -> None:
        from src.schemas.assessment import MasteryLevel, ReasoningScore

        passing = ReasoningScore(
            clinical_accuracy=25,
            reasoning_quality=25,
            terminology=15,
            total=65,
            feedback="Good.",
            mastery_level=MasteryLevel.PROFICIENT,
        )
        failing = ReasoningScore(
            clinical_accuracy=10,
            reasoning_quality=10,
            terminology=5,
            total=25,
            feedback="Needs work.",
            mastery_level=MasteryLevel.NOVICE,
        )
        assert passing.passed is True
        assert failing.passed is False


# ── TopicMastery schema tests ─────────────────────────────────────────────────


class TestTopicMastery:
    def test_update_applies_weighted_average(self) -> None:
        from src.schemas.assessment import TopicMastery

        tm = TopicMastery(topic="cerebellum", score=60.0)
        tm.update(80.0)
        assert 60.0 < tm.score < 80.0

    def test_update_increments_attempts(self) -> None:
        from src.schemas.assessment import TopicMastery

        tm = TopicMastery(topic="hand")
        tm.update(70.0)
        tm.update(80.0)
        assert tm.attempts == 2

    def test_mastery_level_updates_on_score(self) -> None:
        from src.schemas.assessment import MasteryLevel, TopicMastery

        tm = TopicMastery(topic="cerebellum", score=0.0)
        tm.score = 90.0
        tm._update_level()
        assert tm.mastery_level == MasteryLevel.MASTERY


# ── PerformanceSummary tests ──────────────────────────────────────────────────


class TestPerformanceSummary:
    def test_overall_score_calculated(self) -> None:
        from src.schemas.assessment import PerformanceSummary

        summary = PerformanceSummary(
            student_id="s1",
            session_id="sess1",
            mastery_scores={"cerebellum": 80.0, "hand_anatomy": 60.0},
            total_turns=10,
        )
        assert summary.overall_score == 70.0

    def test_empty_scores_returns_zero(self) -> None:
        from src.schemas.assessment import PerformanceSummary

        summary = PerformanceSummary(
            student_id="s2",
            session_id="sess2",
            total_turns=5,
        )
        assert summary.overall_score == 0.0
