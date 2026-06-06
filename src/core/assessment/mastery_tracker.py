"""
src/core/assessment/mastery_tracker.py

MasteryTracker — tracks and updates per-topic mastery scores.
Single responsibility: aggregate scores across turns → PerformanceSummary.
"""

from __future__ import annotations

from src.core.assessment.reasoning_evaluator import ReasoningEvaluator
from src.core.assessment.scenario_generator import ClinicalScenarioGenerator
from src.core.memory.memory_manager import MemoryManager
from src.schemas.assessment import (
    ClinicalScenario,
    PerformanceSummary,
    ReasoningScore,
    TopicMastery,
)
from src.utils.logger import logger


class MasteryTracker:
    """
    Tracks mastery across topics for one student session.

    Args:
        memory_manager: MemoryManager instance (injected)
        evaluator:      ReasoningEvaluator (injected)
        generator:      ClinicalScenarioGenerator (injected)
    """

    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        evaluator: ReasoningEvaluator | None = None,
        generator: ClinicalScenarioGenerator | None = None,
    ) -> None:
        self._memory = memory_manager or MemoryManager()
        self._evaluator = evaluator or ReasoningEvaluator()
        self._generator = generator or ClinicalScenarioGenerator()
        # In-session topic mastery objects
        self._session_mastery: dict[str, TopicMastery] = {}

    async def run_assessment(
        self,
        student_id: str,
        topic: str,
        student_response: str,
    ) -> tuple[ReasoningScore, ClinicalScenario]:
        """
        Run a full assessment turn:
          1. Generate a clinical scenario for the topic
          2. Evaluate the student's response
          3. Update mastery in memory

        Returns:
            (ReasoningScore, ClinicalScenario) tuple
        """
        # Generate scenario
        scenario = await self._generator.generate(topic)

        # Evaluate student response
        score = await self._evaluator.evaluate(student_response, scenario)

        # Update in-session mastery
        if topic not in self._session_mastery:
            self._session_mastery[topic] = TopicMastery(topic=topic)
        self._session_mastery[topic].update(float(score.total))

        # Persist to long-term memory
        await self._memory.update_after_turn(
            student_id=student_id,
            topic=topic,
            score=float(score.total),
        )

        logger.info(
            "Assessment: topic={t}, score={s}, level={l}",
            t=topic,
            s=score.total,
            l=score.mastery_level.value,
        )
        return score, scenario

    async def generate_summary(
        self,
        student_id: str,
        session_id: str,
        total_turns: int,
    ) -> PerformanceSummary:
        """
        Generate end-of-session performance summary.

        Args:
            student_id:   Student identifier
            session_id:   Session identifier
            total_turns:  Total turns in session

        Returns:
            PerformanceSummary with scores, recommendations, next steps
        """
        scores = {t: m.score for t, m in self._session_mastery.items()}

        # Also pull from long-term memory
        memory_scores = await self._memory.get_mastery_scores(student_id)
        scores.update(memory_scores)

        weak_areas = [t for t, s in scores.items() if s < 60]
        strong_areas = [t for t, s in scores.items() if s >= 80]

        narrative = self._build_narrative(scores, weak_areas, strong_areas)
        next_steps = self._build_next_steps(weak_areas, strong_areas)

        # Save session end
        await self._memory.save_session_end(student_id, total_turns)

        return PerformanceSummary(
            student_id=student_id,
            session_id=session_id,
            topics_covered=list(self._session_mastery.keys()),
            mastery_scores=scores,
            weak_areas=weak_areas,
            strong_areas=strong_areas,
            total_turns=total_turns,
            narrative=narrative,
            next_steps=next_steps,
        )

    def _build_narrative(
        self,
        scores: dict[str, float],
        weak: list[str],
        strong: list[str],
    ) -> str:
        if not scores:
            return "Session complete. No scored topics this session."

        avg = sum(scores.values()) / len(scores)

        if avg >= 80:
            opening = "Excellent session — you demonstrated strong understanding."
        elif avg >= 60:
            opening = "Good progress this session with room to develop further."
        else:
            opening = "This session identified areas that need more practice."

        parts = [opening]
        if strong:
            parts.append(f"Strongest areas: {', '.join(t.replace('_', ' ') for t in strong[:2])}.")
        if weak:
            parts.append(
                f"Priority for revision: {', '.join(t.replace('_', ' ') for t in weak[:2])}."
            )
        return " ".join(parts)

    def _build_next_steps(
        self,
        weak: list[str],
        strong: list[str],
    ) -> list[str]:
        steps = []
        for topic in weak[:3]:
            steps.append(
                f"Review {topic.replace('_', ' ')} — focus on clinical implications for OT"
            )
        if strong:
            steps.append(
                f"Deepen knowledge of {strong[0].replace('_', ' ')} "
                f"with advanced clinical scenarios"
            )
        if not steps:
            steps.append("Continue exploring new anatomy topics in the next session")
        return steps
