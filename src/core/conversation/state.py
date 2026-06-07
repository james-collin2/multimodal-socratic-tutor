"""
src/core/conversation/state.py

Conversation state definitions for socratOT.
Single responsibility: define all state types and turn tracking logic.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

# Single source of truth for phase display used in both sidebar and chat topbar.
# Format: phase_value → (label, badge_bg, badge_fg)
PHASE_CONFIG: dict[str, tuple[str, str, str]] = {
    "rapport":    ("Building rapport",     "rgba(56,135,230,.16)",  "#7DB4F5"),
    "tutoring":   ("Socratic tutoring",    "rgba(99,102,241,.18)",  "#A5B4FC"),
    "assessment": ("Clinical assessment",  "rgba(244,150,90,.16)",  "#F6B07D"),
    "mastery":    ("Mastery summary",      "rgba(52,211,153,.16)",  "#6EE7B7"),
}


class ConversationPhase(str, Enum):
    """
    Four sequential phases of every socratOT session.
    Transitions: RAPPORT → TUTORING → ASSESSMENT → MASTERY
    """

    RAPPORT = "rapport"
    TUTORING = "tutoring"
    ASSESSMENT = "assessment"
    MASTERY = "mastery"

    @property
    def display(self) -> str:
        return {
            "rapport": "Building rapport",
            "tutoring": "Socratic tutoring",
            "assessment": "Clinical assessment",
            "mastery": "Mastery summary",
        }[self.value]

    @property
    def color(self) -> str:
        return {
            "rapport": "#185FA5",
            "tutoring": "#534AB7",
            "assessment": "#993C1D",
            "mastery": "#3B6D11",
        }[self.value]

    @property
    def description(self) -> str:
        return {
            "rapport": "Getting to know you",
            "tutoring": "Guided questioning — no direct answers yet",
            "assessment": "Clinical reasoning scenarios",
            "mastery": "Session complete — reviewing progress",
        }[self.value]

    def next_phase(self) -> ConversationPhase | None:
        order = [
            ConversationPhase.RAPPORT,
            ConversationPhase.TUTORING,
            ConversationPhase.ASSESSMENT,
            ConversationPhase.MASTERY,
        ]
        idx = order.index(self)
        return order[idx + 1] if idx < len(order) - 1 else None


class HintLevel(int, Enum):
    """Progressive hint levels within the TUTORING phase."""

    NONE = 0  # ask guiding question, no hint yet
    LEVEL_1 = 1  # broad guiding question
    LEVEL_2 = 2  # narrower clue
    REVEALED = 3  # answer has been revealed


class ResponseQuality(str, Enum):
    """Classification of a student response."""

    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    BYPASS_ATTEMPT = "bypass_attempt"
    OFF_TOPIC = "off_topic"
    UNCLEAR = "unclear"

    @property
    def is_passing(self) -> bool:
        return self in (ResponseQuality.CORRECT, ResponseQuality.PARTIAL)


class TurnRecord(NamedTuple):
    """Immutable record of one conversation turn."""

    turn_number: int
    student_input: str
    tutor_response: str
    phase: ConversationPhase
    hint_level: HintLevel
    response_quality: ResponseQuality | None
    topic: str | None
