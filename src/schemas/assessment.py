"""
src/schemas/assessment.py

Pydantic v2 schemas for assessment engine.
Updated Phase 4: added PerformanceSummary fields.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from src.schemas.base import BaseSchema


class MasteryLevel(str, Enum):
    NOVICE = "novice"
    DEVELOPING = "developing"
    PROFICIENT = "proficient"
    MASTERY = "mastery"


class ClinicalScenario(BaseSchema):
    """A generated clinical OT scenario for assessment."""

    topic: str
    scenario_text: str
    question: str
    reference_answer: str
    difficulty: str = "intermediate"
    ot_context: str | None = None


class ReasoningScore(BaseSchema):
    """Evaluation result for one student response."""

    clinical_accuracy: int = Field(ge=0, le=40)
    reasoning_quality: int = Field(ge=0, le=40)
    terminology: int = Field(ge=0, le=20)
    total: int = Field(ge=0, le=100)
    feedback: str
    mastery_level: MasteryLevel

    @property
    def passed(self) -> bool:
        return self.total >= 60

    @property
    def grade_label(self) -> str:
        if self.total >= 85:
            return "Excellent"
        if self.total >= 70:
            return "Good"
        if self.total >= 60:
            return "Pass"
        return "Needs work"


class TopicMastery(BaseSchema):
    """Per-topic mastery tracking for one student."""

    topic: str
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    attempts: int = Field(default=0, ge=0)
    last_score: float | None = None
    mastery_level: MasteryLevel = MasteryLevel.NOVICE

    def update(self, new_score: float) -> None:
        """Exponential weighted average — 70% existing + 30% new."""
        self.last_score = new_score
        self.attempts += 1
        self.score = round(self.score * 0.7 + new_score * 0.3, 1)
        self._update_level()

    def _update_level(self) -> None:
        if self.score >= 85:
            self.mastery_level = MasteryLevel.MASTERY
        elif self.score >= 70:
            self.mastery_level = MasteryLevel.PROFICIENT
        elif self.score >= 50:
            self.mastery_level = MasteryLevel.DEVELOPING
        else:
            self.mastery_level = MasteryLevel.NOVICE


class PerformanceSummary(BaseSchema):
    """End-of-session performance summary."""

    student_id: str
    session_id: str
    topics_covered: list[str] = Field(default_factory=list)
    mastery_scores: dict[str, float] = Field(default_factory=dict)
    weak_areas: list[str] = Field(default_factory=list)
    strong_areas: list[str] = Field(default_factory=list)
    total_turns: int = Field(ge=0)
    narrative: str = ""
    next_steps: list[str] = Field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.mastery_scores:
            return 0.0
        return round(sum(self.mastery_scores.values()) / len(self.mastery_scores), 1)
