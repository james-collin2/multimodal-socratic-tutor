"""
src/core/assessment/reasoning_evaluator.py

ReasoningEvaluator — scores student clinical reasoning 0–100.
Single responsibility: (student_response, scenario) → ReasoningScore.

Scoring rubric:
  Clinical accuracy  0–40  (correct anatomy + pathology)
  Reasoning quality  0–40  (logical argument, OT context)
  Terminology        0–20  (correct anatomical terms used)
  Total              0–100
"""

from __future__ import annotations

import asyncio

from src.schemas.assessment import ClinicalScenario, MasteryLevel, ReasoningScore
from src.utils.helpers import safe_parse_json, truncate_text
from src.utils.logger import logger


class ReasoningEvaluator:
    """
    LLM-as-judge evaluator for clinical reasoning responses.

    Args:
        llm: Optional LLM (injected for testing)
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
        scenario: ClinicalScenario,
    ) -> ReasoningScore:
        """
        Score a student's clinical reasoning response.

        Args:
            student_response: Student's answer to the scenario question
            scenario:         The ClinicalScenario being assessed

        Returns:
            ReasoningScore with sub-scores and actionable feedback
        """
        if not student_response.strip():
            return ReasoningScore(
                clinical_accuracy=0,
                reasoning_quality=0,
                terminology=0,
                total=0,
                feedback="No response provided.",
                mastery_level=MasteryLevel.NOVICE,
            )

        prompt = f"""You are an expert OT clinical educator scoring a student's response.

Clinical scenario: {scenario.scenario_text}
Question asked: {scenario.question}
Reference answer: {truncate_text(scenario.reference_answer, 400)}
Student response: {truncate_text(student_response, 400)}

Score the student on three dimensions and respond ONLY with valid JSON:
{{
  "clinical_accuracy": 0-40,
  "reasoning_quality": 0-40,
  "terminology": 0-20,
  "feedback": "2-3 sentences of constructive feedback",
  "correct_parts": "what the student got right",
  "missing_parts": "key concepts missing"
}}

Scoring guide:
  clinical_accuracy (40): correct anatomy, correct pathophysiology, correct OT implications
  reasoning_quality (40): logical argument, OT relevance, patient-centred approach
  terminology (20): correct anatomical and OT terminology used accurately"""

        try:
            llm = self._get_llm()
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            raw = resp.content if hasattr(resp, "content") else str(resp)
            data = safe_parse_json(raw)

            if data:
                accuracy = min(40, max(0, int(data.get("clinical_accuracy", 20))))
                reasoning = min(40, max(0, int(data.get("reasoning_quality", 20))))
                terms = min(20, max(0, int(data.get("terminology", 10))))
                total = accuracy + reasoning + terms

                return ReasoningScore(
                    clinical_accuracy=accuracy,
                    reasoning_quality=reasoning,
                    terminology=terms,
                    total=total,
                    feedback=str(data.get("feedback", "")),
                    mastery_level=self._score_to_level(total),
                )
        except Exception as e:
            logger.error("Reasoning evaluation failed: {e}", e=str(e))

        # Fallback: keyword-based scoring
        return self._keyword_fallback(student_response, scenario)

    def _keyword_fallback(
        self,
        response: str,
        scenario: ClinicalScenario,
    ) -> ReasoningScore:
        """Simple keyword overlap scoring when LLM fails."""
        import re

        stopwords = {"the", "a", "an", "is", "are", "to", "of", "in", "and", "or"}

        def kws(text: str) -> set:
            return {w for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()) if w not in stopwords}

        ref_kw = kws(scenario.reference_answer)
        resp_kw = kws(response)
        overlap = len(ref_kw & resp_kw) / max(len(ref_kw), 1)

        accuracy = int(overlap * 40)
        reasoning = int(overlap * 30)
        terms = int(overlap * 20)
        total = accuracy + reasoning + terms

        return ReasoningScore(
            clinical_accuracy=accuracy,
            reasoning_quality=reasoning,
            terminology=terms,
            total=total,
            feedback="Evaluation based on keyword matching — LLM unavailable.",
            mastery_level=self._score_to_level(total),
        )

    @staticmethod
    def _score_to_level(total: int) -> MasteryLevel:
        if total >= 85:
            return MasteryLevel.MASTERY
        if total >= 70:
            return MasteryLevel.PROFICIENT
        if total >= 50:
            return MasteryLevel.DEVELOPING
        return MasteryLevel.NOVICE
