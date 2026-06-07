"""
src/schemas/conversation.py

Pydantic v2 schemas for conversation state, messages, and session data.
Used by ConversationManager, SessionStore, and the Streamlit UI.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from src.schemas.base import BaseSchema, IdentifiedSchema

# ── Enums ─────────────────────────────────────────────────────────────────────


class ConversationPhase(str, Enum):
    """The 4 phases of a tutoring session."""

    RAPPORT = "rapport"
    TUTORING = "tutoring"
    ASSESSMENT = "assessment"
    MASTERY = "mastery"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ResponseQuality(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    UNCLEAR = "unclear"


# ── Message schemas ───────────────────────────────────────────────────────────


class Message(BaseSchema):
    """Single message in a conversation."""

    role: MessageRole
    content: str = Field(min_length=1)
    turn_number: int = Field(ge=0)
    phase: ConversationPhase = ConversationPhase.RAPPORT
    has_image: bool = False
    retrieved_context: str | None = None
    response_quality: ResponseQuality | None = None


class ConversationHistory(BaseSchema):
    """Full message history for a session."""

    messages: list[Message] = Field(default_factory=list)

    def add(self, message: Message) -> None:
        self.messages.append(message)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == MessageRole.USER)

    @property
    def last_user_message(self) -> Message | None:
        for m in reversed(self.messages):
            if m.role == MessageRole.USER:
                return m
        return None


# ── Session schemas ───────────────────────────────────────────────────────────


class StudentProfile(BaseSchema):
    """Lightweight student identity captured in rapport phase."""

    student_id: str = Field(min_length=1)
    name: str | None = None
    program_semester: int | None = Field(default=None, ge=1, le=8)
    preferred_topics: list[str] = Field(default_factory=list)


class SessionState(IdentifiedSchema):
    """
    Complete state of one tutoring session.
    Persisted to SQLite via SessionStore.
    """

    student_id: str
    phase: ConversationPhase = ConversationPhase.RAPPORT
    turn_count: int = Field(default=0, ge=0)
    hint_turns_used: int = Field(default=0, ge=0)
    current_topic: str | None = None
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    is_active: bool = True

    @property
    def answer_reveal_eligible(self) -> bool:
        """True when the student has received enough hints."""
        from src.config.settings import get_settings

        settings = get_settings()
        return self.hint_turns_used >= settings.max_hint_turns

    @property
    def phase_display(self) -> str:
        return self.phase.value.capitalize()
