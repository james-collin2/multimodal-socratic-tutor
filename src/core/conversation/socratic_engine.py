"""
src/core/conversation/socratic_engine.py

SocraticEngine — the core tutor-not-teller logic.
Updated in Phase 4 to support image input via MultimodalPipeline.

Pipeline (text):
  student_input → bypass check → RAG → KnowledgeMasker → LLM → SocraticResponse

Pipeline (image):
  image_bytes → VisionAnalyzer → ImageQuestionGenerator → RAG → SocraticResponse
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.config.settings import get_settings
from src.core.conversation.state import ConversationPhase, HintLevel
from src.core.rag.pipeline import RAGPipeline
from src.prompts.loader import get_prompt
from src.utils.helpers import detect_bypass_attempt, truncate_text
from src.utils.logger import logger


@dataclass
class SocraticResponse:
    """Result from one SocraticEngine call."""

    content: str
    hint_level: HintLevel
    is_bypass_redirect: bool
    is_reveal: bool
    retrieved_context: str
    citations: list[str]
    topic_detected: str | None
    is_image_response: bool = False
    identified_structures: list[str] = field(default_factory=list)


class SocraticEngine:
    """
    Generates Socratic tutoring responses for text and image inputs.

    Args:
        rag_pipeline: RAGPipeline (injected)
        llm:          LangChain LLM (injected)
    """

    BYPASS_RESPONSE = (
        "I understand you want the answer directly — that's natural! "
        "But research shows you retain knowledge much better when you "
        "work through it yourself. Let me guide you with another hint: {hint}"
    )

    def __init__(
        self,
        rag_pipeline: RAGPipeline | None = None,
        llm: object | None = None,
    ) -> None:
        self._rag = rag_pipeline or RAGPipeline()
        self._llm = llm
        self._settings = get_settings()
        self._multimodal = None  # lazy-loaded

    def _get_llm(self) -> object:
        if self._llm is None:
            from src.models.llm_factory import get_llm

            self._llm = get_llm()
        return self._llm

    def _get_multimodal(self) -> object:
        """Lazily load MultimodalPipeline."""
        if self._multimodal is None:
            from src.core.multimodal.pipeline import MultimodalPipeline

            self._multimodal = MultimodalPipeline(rag_pipeline=self._rag)
        return self._multimodal

    # ── Image input entry point ───────────────────────────────────────────────

    async def generate_from_image(
        self,
        image_bytes: bytes,
        media_type: str = "jpeg",
        session_state: dict | None = None,
        user_question: str | None = None,
    ) -> SocraticResponse:
        """
        Generate Socratic response for an uploaded anatomy image.
        Routes through MultimodalPipeline. If the student typed a question
        alongside the image, it is answered guidingly.
        """
        logger.info("SocraticEngine: image input ({n} bytes)", n=len(image_bytes))

        pipeline = self._get_multimodal()
        mm_result = await pipeline.process_image(
            image_bytes, media_type, user_question=user_question
        )

        return SocraticResponse(
            content=mm_result.socratic_reply,
            hint_level=HintLevel.NONE,
            is_bypass_redirect=False,
            is_reveal=False,
            retrieved_context=mm_result.rag_context,
            citations=mm_result.citations,
            topic_detected=mm_result.vision_result.region,
            is_image_response=True,
            identified_structures=mm_result.vision_result.structures,
        )

    # ── Text input entry point ────────────────────────────────────────────────

    async def generate(
        self,
        student_input: str,
        session_state: dict,
        phase: ConversationPhase = ConversationPhase.TUTORING,
    ) -> SocraticResponse:
        """Generate appropriate Socratic response for text input."""
        current_hint = HintLevel(session_state.get("hint_level", 0))
        is_bypass = detect_bypass_attempt(student_input)

        if phase == ConversationPhase.RAPPORT:
            return await self._handle_rapport(student_input, session_state)

        if phase == ConversationPhase.MASTERY:
            return await self._handle_mastery(session_state)

        # Tutoring phase — retrieve with a conversation-aware query so that
        # elliptical follow-ups ("how about in the legs") still find the right chunks.
        retrieval_query = self._build_retrieval_query(
            student_input, session_state.get("history", [])
        )
        rag_result = await self._rag.query(retrieval_query)
        context = rag_result.retrieval.assembled_context
        citations = rag_result.retrieval.citations
        topic = self.detect_topic(student_input)

        if is_bypass and not self._answer_reveal_eligible(current_hint):
            content = await self._redirect_bypass(student_input, context, current_hint)
            return SocraticResponse(
                content=content,
                hint_level=current_hint,
                is_bypass_redirect=True,
                is_reveal=False,
                retrieved_context=context,
                citations=citations,
                topic_detected=topic,
            )

        if self._answer_reveal_eligible(current_hint):
            content = await self._reveal_answer(student_input, context)
            return SocraticResponse(
                content=content,
                hint_level=HintLevel.REVEALED,
                is_bypass_redirect=False,
                is_reveal=True,
                retrieved_context=context,
                citations=citations,
                topic_detected=topic,
            )

        content = await self._generate_hint(
            student_input=student_input,
            context=context,
            hint_level=current_hint,
            session_state=session_state,
        )
        # Advance the hint level so the NEXT turn escalates (NONE → 1 → 2 → reveal).
        # Capped at LEVEL_2 so the reveal happens on the following turn — preserving
        # the "no direct answer in the first two turns" rule.
        next_hint = HintLevel(min(int(current_hint) + 1, int(HintLevel.LEVEL_2)))
        return SocraticResponse(
            content=content,
            hint_level=next_hint,
            is_bypass_redirect=False,
            is_reveal=False,
            retrieved_context=context,
            citations=citations,
            topic_detected=topic,
        )

    # ── Phase handlers ────────────────────────────────────────────────────────

    async def _handle_rapport(self, student_input: str, session_state: dict) -> SocraticResponse:
        # The opening greeting is shown by the UI, so here we only warmly
        # acknowledge the student's reply and pivot to a topic — no re-greeting,
        # no looping on "which semester are you in".
        content = await self._llm_call(
            f"""You are socratOT, a warm Socratic anatomy & neuroscience tutor for OT students.
The student just said: "{student_input}"

In 2 short sentences: acknowledge them by name if they gave one, then invite them to
name an anatomy or neuroscience topic to explore — or, if they already named/asked about
a topic, confirm it and say you'll guide them through it with questions.
Ask at most ONE question. Do NOT lecture or give facts yet."""
        )
        return SocraticResponse(
            content=content,
            hint_level=HintLevel.NONE,
            is_bypass_redirect=False,
            is_reveal=False,
            retrieved_context="",
            citations=[],
            topic_detected=self.detect_topic(student_input),
        )

    async def _handle_mastery(self, session_state: dict) -> SocraticResponse:
        history = session_state.get("history", [])
        topics = list({t["content"] for t in history if t.get("role") == "user" and self.detect_topic(t.get("content", ""))})
        content = get_prompt(
            "socratic.mastery_summary",
            mastered_topics=", ".join(topics) if topics else "anatomy topics",
            weak_topics="Review sessions will be suggested next time.",
        )
        return SocraticResponse(
            content=content,
            hint_level=HintLevel.REVEALED,
            is_bypass_redirect=False,
            is_reveal=True,
            retrieved_context="",
            citations=[],
            topic_detected=None,
        )

    # ── Hint generation ───────────────────────────────────────────────────────

    async def _generate_hint(
        self,
        student_input: str,
        context: str,
        hint_level: HintLevel,
        session_state: dict,
    ) -> str:
        turn = session_state.get("turn_count", 0)
        max_hints = self._settings.max_hint_turns
        masked = get_prompt(
            "socratic.knowledge_mask",
            topic=truncate_text(student_input, 80),
            context=truncate_text(context, 1500),
            current_turn=turn + 1,
            max_hint_turns=max_hints,
        )
        if hint_level == HintLevel.NONE:
            level_instr = (
                "This is your FIRST hint. Ask ONE focused, open-ended question that "
                "nudges the student toward the answer, anchored in a specific detail "
                "from the context above. Do NOT state any fact from the answer "
                "directly. Keep it to 1–2 sentences."
            )
        else:
            level_instr = (
                "This is your SECOND, STRONGER hint. Point to the specific concept or "
                "term in the context that leads to the answer and ask a pointed "
                "question about it — but still do NOT give the full answer. "
                "Keep it to 1–2 sentences."
            )
        prompt = (
            f"{masked}\n\n"
            f'Student question: "{student_input}"\n\n'
            f"{level_instr}\n"
            f"Respond with ONLY the Socratic hint/question — no preamble, no greeting."
        )
        return await self._llm_call(prompt)

    async def _redirect_bypass(
        self, student_input: str, context: str, hint_level: HintLevel
    ) -> str:
        hint = get_prompt(
            "socratic.hint_level_1",
            topic=truncate_text(student_input, 60),
            guiding_question="What clues from the context help you think through this?",
        )
        return self.BYPASS_RESPONSE.format(hint=hint[:200])

    async def _reveal_answer(self, student_input: str, context: str) -> str:
        prompt = get_prompt(
            "socratic.reveal_after_turns",
            direct_answer=f"Based on the context: {truncate_text(context, 800)}",
        )
        return await self._llm_call(
            f"{prompt}\n\nStudent question: {student_input}\n"
            f"Context: {truncate_text(context, 1000)}\n\n"
            f"Provide a clear complete answer now. Reference the source material."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _answer_reveal_eligible(self, hint_level: HintLevel) -> bool:
        return hint_level >= HintLevel.LEVEL_2

    def _build_retrieval_query(self, student_input: str, history: list) -> str:
        """
        Make elliptical follow-ups retrievable by prepending the most recent
        substantive user question. Self-contained questions (those with a clear
        topic) are used as-is.
        """
        if self.detect_topic(student_input):
            return student_input
        t = student_input.lower().strip()
        cues = ("how about", "what about", "and ", "what else", "what of", "and the")
        if (len(t.split()) <= 5 or t.startswith(cues)) and history:
            for h in reversed(history):
                if h.get("role") == "user" and len(h.get("content", "").split()) >= 3:
                    return f"{h['content']} {student_input}".strip()
        return student_input

    def detect_topic(self, student_input: str) -> str | None:
        keywords = {
            "cerebellum": "cerebellum",
            "cranial": "cranial_nerves",
            "facial nerve": "cranial_nerves",
            "median nerve": "peripheral_nervous_system",
            "ulnar": "peripheral_nervous_system",
            "radial nerve": "peripheral_nervous_system",
            "brachial": "peripheral_nervous_system",
            "hand": "hand_anatomy",
            "carpal": "hand_anatomy",
            "spinal cord": "spinal_cord",
            "dermatome": "spinal_cord",
            "basal ganglia": "basal_ganglia",
        }
        lower = student_input.lower()
        for kw, topic in keywords.items():
            if kw in lower:
                return topic
        return None

    async def _llm_call(self, prompt: str) -> str:
        try:
            llm = self._get_llm()
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            return (resp.content if hasattr(resp, "content") else str(resp)).strip()
        except Exception as e:
            logger.error("LLM call failed: {e}", e=str(e))
            return (
                "I'm having trouble connecting to the AI model right now. "
                "Please check your OpenAI API key and try again."
            )
