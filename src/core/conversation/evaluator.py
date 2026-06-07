"""
src/core/conversation/evaluator.py

StudentResponseEvaluator — classifies student answers and generates feedback.
Single responsibility: given (student_response, reference, context),
return an EvaluationResult with quality classification and feedback.

Uses LLM-as-judge approach: structured JSON output from gpt-4o-mini.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.core.conversation.state import ResponseQuality
from src.utils.helpers import safe_parse_json, truncate_text
from src.utils.logger import logger


@dataclass
class EvaluationResult:
    """Result from evaluating one student response."""

    quality: ResponseQuality
    score: float  # 0.0 – 1.0
    feedback: str  # one sentence of constructive feedback
    correct_parts: str  # what the student got right
    missing_parts: str  # what was missing or wrong
    encouragement: str  # affirmation or redirect message


class StudentResponseEvaluator:
    """
    Evaluates student responses against reference answers.

    Two-stage approach:
      1. Keyword overlap check (fast, no LLM)
      2. LLM-as-judge for semantic evaluation (full quality score)

    Args:
        llm: LangChain LLM instance (injected)
    """

    def __init__(self, llm: object | None = None) -> None:
        self._llm = llm

    def _get_llm(self) -> object:
        if self._llm is None:
            from src.models.llm_factory import get_llm

            self._llm = get_llm()
        return self._llm

    async def evaluate(
        self,
        student_response: str,
        reference_answer: str,
        context: str = "",
        topic: str = "",
    ) -> EvaluationResult:
        """
        Evaluate a student response against the reference answer.

        Args:
            student_response:  What the student said
            reference_answer:  The correct/expected answer
            context:           RAG context used in the conversation
            topic:             Current anatomy topic

        Returns:
            EvaluationResult with quality, score, and feedback
        """
        if not student_response.strip():
            return EvaluationResult(
                quality=ResponseQuality.UNCLEAR,
                score=0.0,
                feedback="Please provide a response to continue.",
                correct_parts="",
                missing_parts="No response given.",
                encouragement="Take your time — what do you think?",
            )

        # Stage 1: fast keyword check
        overlap = self.keyword_overlap(student_response, reference_answer)

        # Stage 2: LLM semantic evaluation
        try:
            result = await self._llm_evaluate(
                student_response=student_response,
                reference_answer=reference_answer,
                context=context,
                topic=topic,
                initial_overlap=overlap,
            )
            return result
        except Exception as e:
            logger.warning("LLM evaluation failed, using keyword overlap: {e}", e=str(e))
            return self._fallback_evaluation(student_response, overlap)

    async def _llm_evaluate(
        self,
        student_response: str,
        reference_answer: str,
        context: str,
        topic: str,
        initial_overlap: float,
    ) -> EvaluationResult:
        """Use LLM judge to evaluate semantic quality."""
        prompt = f"""You are an expert OT anatomy tutor evaluating a student response.

Topic: {topic or "Anatomy/Neuroscience"}

Reference answer: {truncate_text(reference_answer, 400)}

Student response: {truncate_text(student_response, 300)}

Evaluate the student response and return ONLY valid JSON:
{{
  "quality": "correct" | "partial" | "incorrect",
  "score": <float 0.0-1.0>,
  "feedback": "<one sentence of constructive feedback>",
  "correct_parts": "<what the student got right, or empty string>",
  "missing_parts": "<key concepts missing or wrong, or empty string>",
  "encouragement": "<one sentence — affirm correct, redirect partial, reframe incorrect>"
}}"""

        llm = self._get_llm()
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: llm.invoke(prompt),
        )
        text = raw.content if hasattr(raw, "content") else str(raw)
        parsed = safe_parse_json(text)

        if not parsed:
            return self._fallback_evaluation(student_response, initial_overlap)

        quality_str = parsed.get("quality", "unclear").lower()
        quality_map = {
            "correct": ResponseQuality.CORRECT,
            "partial": ResponseQuality.PARTIAL,
            "incorrect": ResponseQuality.INCORRECT,
        }
        quality = quality_map.get(quality_str, ResponseQuality.UNCLEAR)

        return EvaluationResult(
            quality=quality,
            score=float(parsed.get("score", 0.5)),
            feedback=str(parsed.get("feedback", "")),
            correct_parts=str(parsed.get("correct_parts", "")),
            missing_parts=str(parsed.get("missing_parts", "")),
            encouragement=str(parsed.get("encouragement", "")),
        )

    def _fallback_evaluation(
        self,
        student_response: str,
        overlap: float,
    ) -> EvaluationResult:
        """Keyword-based fallback when LLM evaluation fails."""
        if overlap >= 0.6:
            return EvaluationResult(
                quality=ResponseQuality.CORRECT,
                score=overlap,
                feedback="Good answer — key concepts identified.",
                correct_parts=student_response[:100],
                missing_parts="",
                encouragement="Excellent reasoning! Can you elaborate further?",
            )
        elif overlap >= 0.3:
            return EvaluationResult(
                quality=ResponseQuality.PARTIAL,
                score=overlap,
                feedback="Partially correct — some key concepts missing.",
                correct_parts="Some relevant terms identified.",
                missing_parts="Review the full definition.",
                encouragement="You are on the right track — what else do you know?",
            )
        else:
            return EvaluationResult(
                quality=ResponseQuality.INCORRECT,
                score=overlap,
                feedback="Not quite — let us revisit this together.",
                correct_parts="",
                missing_parts="Key concepts not identified.",
                encouragement="That is a common misconception — let me guide you.",
            )

    def keyword_overlap(self, response: str, reference: str) -> float:
        """Calculate keyword overlap ratio between response and reference."""
        import re

        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "and",
            "or",
            "but",
            "not",
            "this",
            "that",
            "it",
            "its",
            "which",
        }

        def keywords(text: str) -> set:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            return {w for w in words if w not in stopwords}

        resp_kw = keywords(response)
        ref_kw = keywords(reference)
        if not resp_kw or not ref_kw:
            return 0.0
        return len(resp_kw & ref_kw) / max(len(resp_kw), 1)
