"""
src/core/rag/hallucination_guard.py

HallucinationGuard — verifies LLM answer stays grounded in retrieved context.
Single responsibility: (answer, context) in → GuardResult out.

Uses simple keyword overlap + LLM-based verification.
Flags answers that introduce facts not present in retrieved context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.helpers import safe_parse_json
from src.utils.logger import logger


@dataclass
class GuardResult:
    """Result from hallucination check."""

    is_grounded: bool
    confidence: float  # 0.0 = definitely hallucinated, 1.0 = fully grounded
    flagged_phrases: list[str]
    reason: str


class HallucinationGuard:
    """
    Checks if an LLM-generated answer is grounded in retrieved context.

    Two-stage approach:
    1. Fast keyword overlap check (no LLM call)
    2. LLM-based semantic verification (when keyword check is ambiguous)

    Args:
        llm: LangChain LLM instance for semantic verification
        overlap_threshold: Min keyword overlap ratio to pass fast check
        use_llm_verification: Whether to use LLM for semantic check
    """

    def __init__(
        self,
        llm: object | None = None,
        overlap_threshold: float = 0.15,
        use_llm_verification: bool = True,
    ) -> None:
        self._llm = llm
        self._overlap_threshold = overlap_threshold
        self._use_llm = use_llm_verification

    async def check(
        self,
        answer: str,
        context: str,
        query: str = "",
    ) -> GuardResult:
        """
        Check if answer is grounded in context.

        Args:
            answer:  LLM-generated answer to verify
            context: Retrieved context the answer should be based on
            query:   Original query (used for LLM verification prompt)

        Returns:
            GuardResult with grounding assessment
        """
        if not context.strip():
            return GuardResult(
                is_grounded=False,
                confidence=0.0,
                flagged_phrases=[],
                reason="No context provided — cannot verify grounding",
            )

        # Stage 1: Fast keyword overlap check
        overlap_score = self._keyword_overlap(answer, context)
        logger.debug("Hallucination guard overlap score: {s:.3f}", s=overlap_score)

        if overlap_score >= self._overlap_threshold:
            return GuardResult(
                is_grounded=True,
                confidence=min(1.0, overlap_score * 2),
                flagged_phrases=[],
                reason=f"Keyword overlap {overlap_score:.2f} above threshold",
            )

        # Stage 2: LLM semantic verification (if enabled and LLM available)
        if self._use_llm and self._llm is not None:
            return await self._llm_verify(answer, context, query)

        # Stage 2 fallback: flag as potentially hallucinated
        return GuardResult(
            is_grounded=overlap_score > 0.05,
            confidence=overlap_score,
            flagged_phrases=self._find_unsupported_phrases(answer, context),
            reason=f"Low keyword overlap ({overlap_score:.2f}) — may contain hallucinations",
        )

    def _keyword_overlap(self, answer: str, context: str) -> float:
        """
        Calculate keyword overlap ratio between answer and context.
        Strips stopwords and measures substantive word overlap.
        """
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
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "and",
            "or",
            "but",
            "not",
            "this",
            "that",
            "it",
            "its",
            "they",
            "them",
            "their",
            "which",
            "who",
            "what",
            "how",
            "when",
            "where",
            "why",
            "all",
            "also",
            "as",
            "into",
        }

        def extract_keywords(text: str) -> set:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            return {w for w in words if w not in stopwords}

        answer_kw = extract_keywords(answer)
        context_kw = extract_keywords(context)

        if not answer_kw:
            return 0.0

        overlap = answer_kw.intersection(context_kw)
        return len(overlap) / len(answer_kw)

    def _find_unsupported_phrases(
        self,
        answer: str,
        context: str,
    ) -> list[str]:
        """
        Find answer phrases that have no support in context.
        Returns up to 3 suspicious phrases.
        """
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 20]
        context_lower = context.lower()
        flagged = []

        for sentence in sentences[:10]:
            # Check if key nouns from sentence appear in context
            nouns = re.findall(r"\b[A-Z][a-z]{3,}\b", sentence)
            if nouns:
                found = any(n.lower() in context_lower for n in nouns)
                if not found:
                    flagged.append(sentence[:80])

        return flagged[:3]

    async def _llm_verify(
        self,
        answer: str,
        context: str,
        query: str,
    ) -> GuardResult:
        """Use LLM to semantically verify answer grounding."""
        prompt = f"""You are a fact-checker for an educational AI system.

Given the following retrieved context and generated answer,
determine if the answer is grounded in the context or introduces
facts not present in the context.

Context:
{context[:2000]}

Question: {query}
Answer: {answer}

Respond ONLY with valid JSON:
{{
  "is_grounded": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}}"""

        try:
            import asyncio

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self._llm.invoke(prompt))
            text = response.content if hasattr(response, "content") else str(response)
            parsed = safe_parse_json(text)

            if parsed:
                return GuardResult(
                    is_grounded=bool(parsed.get("is_grounded", False)),
                    confidence=float(parsed.get("confidence", 0.5)),
                    flagged_phrases=[],
                    reason=str(parsed.get("reason", "LLM verification")),
                )
        except Exception as e:
            logger.warning("LLM verification failed: {e}", e=str(e))

        return GuardResult(
            is_grounded=False,
            confidence=0.3,
            flagged_phrases=[],
            reason="LLM verification failed — treating as unverified",
        )
