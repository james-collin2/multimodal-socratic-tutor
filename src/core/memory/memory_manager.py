"""
src/core/memory/memory_manager.py

MemoryManager — loads memory at session start, saves at session end.
Single responsibility: bridge between StudentMemory and ConversationManager.

Key feature: PersonalisedOpener — generates session opening that
references past weak areas so students feel the system knows them.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.memory.student_memory import MemoryRecord, StudentMemory
from src.utils.logger import logger


@dataclass
class SessionContext:
    """Memory context loaded at session start."""

    record: MemoryRecord
    is_returning: bool  # True if student has prior sessions
    personalised_opener: str  # opening message referencing past work
    priority_topics: list[str]  # weak topics to revisit this session


class MemoryManager:
    """
    Manages student memory across sessions.

    Usage:
        manager = MemoryManager()
        ctx = await manager.load_session_context("student_001", "Alice")
        # use ctx.personalised_opener as first message
        await manager.update_after_turn("student_001", "cerebellum", 72.0)
        await manager.save_session_end("student_001", session_turns=15)
    """

    def __init__(self, memory: StudentMemory | None = None) -> None:
        self._memory = memory or StudentMemory()

    async def load_session_context(
        self,
        student_id: str,
        student_name: str = "",
    ) -> SessionContext:
        """Load memory and build personalised session context."""
        record = await self._memory.load(student_id)
        is_returning = record.total_sessions > 0

        opener = self._build_opener(record, student_name, is_returning)

        # Priority = weak topics ordered by score (lowest first)
        priority = sorted(
            record.weak_topics,
            key=lambda t: record.mastery_scores.get(t, 0.0),
        )[:3]

        logger.info(
            "Memory loaded for {id}: {sessions} sessions, {weak} weak topics",
            id=student_id,
            sessions=record.total_sessions,
            weak=len(record.weak_topics),
        )

        return SessionContext(
            record=record,
            is_returning=is_returning,
            personalised_opener=opener,
            priority_topics=priority,
        )

    def _build_opener(
        self,
        record: MemoryRecord,
        name: str,
        is_returning: bool,
    ) -> str:
        """Build personalised session opener from memory."""
        greeting = f"Welcome back, {name}!" if name else "Welcome back!"

        if not is_returning:
            return (
                "Welcome to socratOT! I am your Socratic anatomy and "
                "neuroscience tutor. I teach through guided questioning — "
                "I will never simply give you the answer, but I will guide "
                "you to discover it yourself. What would you like to explore today?"
            )

        parts = [greeting]

        if record.weak_topics:
            weak_str = ", ".join(t.replace("_", " ") for t in record.weak_topics[:2])
            parts.append(
                f"Last session you found **{weak_str}** challenging — "
                f"would you like to revisit that today, or explore something new?"
            )
        elif record.strong_topics:
            strong_str = record.strong_topics[0].replace("_", " ")
            parts.append(
                f"You have been making great progress with {strong_str}. "
                f"Ready to go deeper or try a new topic?"
            )
        else:
            parts.append("Ready to continue your anatomy journey?")

        return " ".join(parts)

    async def update_after_turn(
        self,
        student_id: str,
        topic: str,
        score: float,
    ) -> None:
        """Update mastery score after a scored turn."""
        await self._memory.update_topic_score(student_id, topic, score)

    async def save_session_end(
        self,
        student_id: str,
        session_turns: int,
    ) -> None:
        """Increment session count and total turns at session end."""
        record = await self._memory.load(student_id)
        record.total_sessions += 1
        record.total_turns += session_turns
        await self._memory.save(record)
        logger.info(
            "Session saved for {id}: {sessions} total sessions",
            id=student_id,
            sessions=record.total_sessions,
        )

    async def get_weak_topics(self, student_id: str) -> list[str]:
        """Return current weak topics for a student."""
        record = await self._memory.load(student_id)
        return record.weak_topics

    async def get_mastery_scores(self, student_id: str) -> dict[str, float]:
        """Return all mastery scores for a student."""
        record = await self._memory.load(student_id)
        return record.mastery_scores
