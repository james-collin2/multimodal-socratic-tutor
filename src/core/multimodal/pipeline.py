"""
src/core/multimodal/pipeline.py

MultimodalPipeline — orchestrates the full image → Socratic response flow.
Single responsibility: image bytes in → SocraticResponse out.

Pipeline:
  image bytes
    → VisionAnalyzer (GPT-4o)
    → ImageQuestionGenerator
    → RAG context retrieval for identified structures
    → SocraticResponse
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.multimodal.question_generator import (
    ImageQuestion,
    ImageQuestionGenerator,
)
from src.core.multimodal.vision_analyzer import VisionAnalyzer, VisionResult
from src.utils.logger import logger


@dataclass
class MultimodalResponse:
    """Complete response from the multimodal pipeline."""

    vision_result: VisionResult
    questions: list[ImageQuestion]
    rag_context: str
    socratic_reply: str  # the tutor's opening message
    citations: list[str]


class MultimodalPipeline:
    """
    Orchestrates image analysis → question generation → RAG grounding.

    Args:
        vision_analyzer:    VisionAnalyzer instance (injected)
        question_generator: ImageQuestionGenerator instance (injected)
        rag_pipeline:       RAGPipeline instance (injected)
    """

    def __init__(
        self,
        vision_analyzer: VisionAnalyzer | None = None,
        question_generator: ImageQuestionGenerator | None = None,
        rag_pipeline: object | None = None,
    ) -> None:
        self._vision = vision_analyzer or VisionAnalyzer()
        self._questions = question_generator or ImageQuestionGenerator()
        self._rag = rag_pipeline

    def _get_rag(self) -> object:
        if self._rag is None:
            from src.core.rag.pipeline import RAGPipeline

            self._rag = RAGPipeline()
        return self._rag

    async def process_image(
        self,
        image_bytes: bytes,
        media_type: str = "jpeg",
        user_question: str | None = None,
    ) -> MultimodalResponse:
        """
        Full pipeline: image bytes → structured Socratic response.

        Args:
            image_bytes:   Raw image data
            media_type:    MIME sub-type (jpeg, png, webp)
            user_question: Optional question the student asked about the image.
                           When given, the Socratic opener answers it (guidingly).

        Returns:
            MultimodalResponse with questions and tutor message
        """
        logger.info("Multimodal pipeline: analyzing image ({n} bytes)", n=len(image_bytes))

        # Step 1: Vision analysis
        vision_result = await self._vision.analyze_bytes(image_bytes, media_type)

        if vision_result.error:
            logger.warning("Vision analysis failed: {e}", e=vision_result.error)
            return self._error_response(vision_result)

        logger.info(
            "Vision: {n} structures in region={r} (confidence={c:.2f})",
            n=len(vision_result.structures),
            r=vision_result.region,
            c=vision_result.confidence,
        )

        # Step 2: Generate Socratic questions
        questions = await self._questions.generate(vision_result, n_questions=3)

        # Step 3: RAG context for identified structures
        rag_context = ""
        citations = []
        if vision_result.structures:
            query = f"{vision_result.region} anatomy {' '.join(vision_result.structures[:3])}"
            try:
                rag_result = await self._get_rag().query(query)
                rag_context = rag_result.retrieval.assembled_context
                citations = rag_result.retrieval.citations
            except Exception as e:
                logger.warning("RAG retrieval failed for image query: {e}", e=str(e))

        # Step 4: Build Socratic opener. If the student asked a question about the
        # image, answer it guidingly (LLM); otherwise use the templated opener.
        if user_question and user_question.strip():
            socratic_reply = await self._build_opener_with_question(
                vision_result, questions, rag_context, user_question.strip()
            )
        else:
            socratic_reply = self._build_opener(vision_result, questions)

        return MultimodalResponse(
            vision_result=vision_result,
            questions=questions,
            rag_context=rag_context,
            socratic_reply=socratic_reply,
            citations=citations,
        )

    def _build_opener(
        self,
        vision: VisionResult,
        questions: list[ImageQuestion],
    ) -> str:
        """Build the tutor's opening message for an uploaded image."""
        structures_str = (
            ", ".join(vision.structures[:4])
            if vision.structures
            else "several anatomical structures"
        )

        first_q = (
            questions[0].question
            if questions
            else "What structures can you identify and what do they do?"
        )

        return (
            f"I can see an anatomy image showing **{structures_str}** "
            f"in the **{vision.region.replace('_', ' ')}** region. "
            f"Rather than describing everything, let me guide you to discover it. "
            f"\n\n{first_q}"
        )

    async def _build_opener_with_question(
        self,
        vision: VisionResult,
        questions: list[ImageQuestion],
        rag_context: str,
        user_question: str,
    ) -> str:
        """
        Build a Socratic reply that addresses the student's question about the
        image — guiding, not telling. Falls back to the templated opener if the
        LLM is unavailable (blind-test safety).
        """
        structures_str = ", ".join(vision.structures[:5]) or "the structures shown"
        prompt = (
            "You are socratOT, a Socratic anatomy & neuroscience tutor for OT students.\n"
            f'The student uploaded an anatomy image and asked: "{user_question}"\n\n'
            f"Vision analysis — region: {vision.region}; "
            f"structures identified: {structures_str}.\n"
            f"Reference context:\n{(rag_context or 'N/A')[:1200]}\n\n"
            "Respond in 2-3 sentences. Do NOT give the full answer directly. "
            "Acknowledge what is visible, then ask ONE focused guiding question that "
            "leads the student toward answering their own question. No preamble."
        )
        try:
            import asyncio

            from src.models.llm_factory import get_llm

            llm = get_llm()
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            if text:
                return text
        except Exception as e:
            logger.warning("Image-question opener LLM failed, using template: {e}", e=str(e))

        # Fallback: templated opener + acknowledge the question
        base = self._build_opener(vision, questions)
        return f"You asked: *{user_question}*\n\n{base}"

    def _error_response(self, vision: VisionResult) -> MultimodalResponse:
        return MultimodalResponse(
            vision_result=vision,
            questions=[],
            rag_context="",
            socratic_reply=(
                "I had trouble analysing that image. "
                "Please ensure it is a clear anatomy diagram and try again. "
                "You can also describe what you see in the image and I will guide you."
            ),
            citations=[],
        )
