"""
src/core/conversation/manager.py

ConversationManager — orchestrates all four session phases.
Updated in Phase 4 to wire in MemoryManager and MasteryTracker.
"""

from __future__ import annotations

from src.config.settings import get_settings
from src.core.conversation.session_store import SessionStore
from src.core.conversation.state import (
    ConversationPhase,
    HintLevel,
    ResponseQuality,
    TurnRecord,
)
from src.utils.exceptions import InvalidPhaseTransitionError, SessionNotFoundError
from src.utils.helpers import detect_bypass_attempt, sanitize_student_id
from src.utils.logger import logger


class ConversationManager:
    """
    Manages full lifecycle of a socratOT session.
    Phase 4: wired to MemoryManager for cross-session memory.
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        memory_manager: object | None = None,
    ) -> None:
        self._store = store or SessionStore()
        self._memory = memory_manager
        self._settings = get_settings()

    def _get_memory(self) -> object:
        if self._memory is None:
            from src.core.memory.memory_manager import MemoryManager

            self._memory = MemoryManager()
        return self._memory

    # ── Session lifecycle ──────────────────────────────────────────────────────

    async def start_session(
        self,
        student_id: str,
        student_name: str = "",
    ) -> tuple[str, str]:
        """
        Create a new session. Returns (session_id, personalised_opener).
        Phase 4: loads memory and generates personalised opener.
        """
        safe_id = sanitize_student_id(student_id)
        session_id = await self._store.create_session(safe_id)

        # Load memory for personalised opener
        try:
            ctx = await self._get_memory().load_session_context(safe_id, student_name)
            opener = ctx.personalised_opener
        except Exception as e:
            logger.warning("Memory load failed: {e}", e=str(e))
            opener = (
                "Welcome to socratOT! I teach through guided questioning — "
                "I will never simply give you the answer, but I will guide "
                "you to discover it yourself. What would you like to explore today?"
            )

        logger.info("Session started: {sid} for {uid}", sid=session_id[:8], uid=safe_id)
        return session_id, opener

    async def get_state(self, session_id: str) -> dict:
        return await self._store.get_session(session_id)

    async def end_session(self, session_id: str) -> None:
        state = await self._store.get_session(session_id)
        turns = state.get("turn_count", 0)
        s_id = state.get("student_id", "")

        await self._store.update_session(
            session_id,
            phase=ConversationPhase.MASTERY,
            is_active=False,
        )

        # Save to long-term memory
        try:
            await self._get_memory().save_session_end(s_id, turns)
        except Exception as e:
            logger.warning("Memory save failed at session end: {e}", e=str(e))

        logger.info("Session ended: {sid}", sid=session_id[:8])

    # ── Turn management ────────────────────────────────────────────────────────

    async def advance_turn(
        self,
        session_id: str,
        student_input: str,
        tutor_response: str,
        response_quality: ResponseQuality | None = None,
        topic: str | None = None,
        mastery_score: float | None = None,
    ) -> dict:
        """Record one completed turn. Phase 4: optionally updates memory score."""
        state = await self._store.get_session(session_id)
        current_phase = ConversationPhase(state["phase"])
        current_turn = state["turn_count"] + 1
        current_hint = HintLevel(state.get("hint_level", 0))

        is_bypass = detect_bypass_attempt(student_input)
        if is_bypass:
            response_quality = ResponseQuality.BYPASS_ATTEMPT
            logger.warning(
                "Bypass attempt: session={sid} turn={t}", sid=session_id[:8], t=current_turn
            )

        # Advance hint level
        new_hint = current_hint
        if current_phase == ConversationPhase.TUTORING:
            if current_hint == HintLevel.NONE:
                new_hint = HintLevel.LEVEL_1
            elif current_hint == HintLevel.LEVEL_1:
                new_hint = HintLevel.LEVEL_2
            elif current_hint == HintLevel.LEVEL_2:
                new_hint = HintLevel.REVEALED

        record = TurnRecord(
            turn_number=current_turn,
            student_input=student_input,
            tutor_response=tutor_response,
            phase=current_phase,
            hint_level=new_hint,
            response_quality=response_quality,
            topic=topic or state.get("current_topic"),
        )

        history = state.get("history", [])
        history.append(
            {
                "turn": record.turn_number,
                "student": record.student_input,
                "tutor": record.tutor_response,
                "phase": record.phase.value,
                "hint_level": int(record.hint_level),
                "quality": record.response_quality.value if record.response_quality else None,
                "topic": record.topic,
            }
        )

        await self._store.update_session(
            session_id,
            turn_count=current_turn,
            hint_level=new_hint,
            current_topic=topic or state.get("current_topic"),
            history=history,
        )

        # Update memory if score provided
        if mastery_score is not None and topic:
            student_id = state.get("student_id", "")
            try:
                await self._get_memory().update_after_turn(student_id, topic, mastery_score)
            except Exception as e:
                logger.warning("Memory update failed: {e}", e=str(e))

        return await self._store.get_session(session_id)

    # ── Phase transitions ──────────────────────────────────────────────────────

    async def transition_phase(self, session_id: str, target: ConversationPhase) -> None:
        state = await self._store.get_session(session_id)
        current = ConversationPhase(state["phase"])
        if target == current:
            return
        valid_next = current.next_phase()
        if target != valid_next:
            raise InvalidPhaseTransitionError(
                f"Cannot transition from {current.value} to {target.value}",
                detail=f"Expected: {valid_next}",
            )
        reset_hint = HintLevel.NONE if target == ConversationPhase.TUTORING else None
        await self._store.update_session(session_id, phase=target, hint_level=reset_hint)
        logger.info(
            "Phase: {old} → {new} ({sid})", old=current.value, new=target.value, sid=session_id[:8]
        )

    async def transition_to_tutoring(self, session_id: str) -> None:
        await self.transition_phase(session_id, ConversationPhase.TUTORING)

    async def transition_to_assessment(self, session_id: str) -> None:
        await self.transition_phase(session_id, ConversationPhase.ASSESSMENT)

    async def transition_to_mastery(self, session_id: str) -> None:
        await self.transition_phase(session_id, ConversationPhase.MASTERY)

    # ── Eligibility ───────────────────────────────────────────────────────────

    async def answer_reveal_eligible(self, session_id: str) -> bool:
        state = await self._store.get_session(session_id)
        return HintLevel(state.get("hint_level", 0)) >= HintLevel.LEVEL_2

    async def should_transition_to_assessment(self, session_id: str) -> bool:
        state = await self._store.get_session(session_id)
        return state["turn_count"] >= self._settings.max_session_turns // 2

    async def is_session_active(self, session_id: str) -> bool:
        try:
            return bool((await self._store.get_session(session_id)).get("is_active", 1))
        except SessionNotFoundError:
            return False

    async def get_student_memory(self, student_id: str) -> dict:
        """Return student's long-term memory as dict for UI display."""
        try:
            return await self._get_memory().get_mastery_scores(student_id)
        except Exception:
            return {}
