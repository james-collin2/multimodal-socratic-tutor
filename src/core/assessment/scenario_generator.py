"""
src/core/assessment/scenario_generator.py

ClinicalScenarioGenerator — creates OT clinical scenarios from RAG corpus.
Single responsibility: topic → ClinicalScenario with OT context.
"""

from __future__ import annotations

import asyncio

from src.schemas.assessment import ClinicalScenario
from src.utils.helpers import safe_parse_json, truncate_text
from src.utils.logger import logger

# OT-specific scenario templates per topic
SCENARIO_TEMPLATES = {
    "cerebellum": (
        "A 45-year-old patient presents to OT following a posterior fossa "
        "stroke affecting the cerebellum. They report difficulty with "
        "self-care tasks and their family notes they spill drinks frequently."
    ),
    "cranial_nerves": (
        "A 62-year-old patient is referred for OT following Bell's palsy "
        "affecting the right facial nerve. They are struggling with eating, "
        "oral hygiene, and social participation."
    ),
    "peripheral_nervous_system": (
        "A 38-year-old office worker presents with numbness and tingling in "
        "the thumb, index, and middle fingers after prolonged computer use. "
        "Thenar wasting is noted on examination."
    ),
    "hand_anatomy": (
        "A 55-year-old carpenter presents following a zone II flexor tendon "
        "laceration to the right index finger. They are 4 weeks post-surgery "
        "and starting active range of motion."
    ),
    "spinal_cord": (
        "A 28-year-old presents to inpatient OT following a C6 complete "
        "spinal cord injury from a diving accident. Goals include maximising "
        "upper limb function and independence in ADLs."
    ),
}

GENERIC_SCENARIO = (
    "A patient has been referred to occupational therapy following a "
    "neurological injury affecting their ability to perform activities "
    "of daily living independently."
)


class ClinicalScenarioGenerator:
    """
    Generates OT clinical scenarios using LLM + RAG context.

    Args:
        llm: Optional LLM (injected)
        rag_pipeline: Optional RAGPipeline (injected)
    """

    def __init__(
        self,
        llm: object | None = None,
        rag_pipeline: object | None = None,
    ) -> None:
        self._llm = llm
        self._rag = rag_pipeline

    def _get_llm(self) -> object:
        if self._llm is None:
            from src.models.llm_factory import get_llm

            self._llm = get_llm()
        return self._llm

    def _get_rag(self) -> object:
        if self._rag is None:
            from src.core.rag.pipeline import RAGPipeline

            self._rag = RAGPipeline()
        return self._rag

    async def generate(
        self,
        topic: str,
        difficulty: str = "intermediate",
    ) -> ClinicalScenario:
        """
        Generate a clinical OT scenario for a given topic.

        Args:
            topic:      Anatomy topic (e.g. 'cerebellum', 'spinal_cord')
            difficulty: beginner | intermediate | advanced

        Returns:
            ClinicalScenario with scenario_text, question, and reference_answer
        """
        # Get RAG context for this topic
        context = ""
        try:
            rag_result = await self._get_rag().query(
                f"{topic.replace('_', ' ')} occupational therapy clinical"
            )
            context = rag_result.retrieval.assembled_context
        except Exception as e:
            logger.warning("RAG failed for scenario: {e}", e=str(e))

        template = SCENARIO_TEMPLATES.get(topic, GENERIC_SCENARIO)

        prompt = f"""You are an OT clinical educator. Generate ONE clinical assessment scenario.

Topic: {topic.replace("_", " ")}
Difficulty: {difficulty}
Base scenario: {template}
Anatomy context: {truncate_text(context, 800)}

Generate a clinical scenario and respond ONLY with valid JSON:
{{
  "scenario_text": "detailed 3-4 sentence clinical presentation",
  "question": "one specific clinical reasoning question for an OT student",
  "reference_answer": "comprehensive 3-4 sentence reference answer",
  "ot_context": "one sentence on specific OT intervention focus"
}}"""

        try:
            llm = self._get_llm()
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            raw = resp.content if hasattr(resp, "content") else str(resp)
            data = safe_parse_json(raw)

            if data:
                return ClinicalScenario(
                    topic=topic,
                    scenario_text=data.get("scenario_text", template),
                    question=data.get(
                        "question",
                        f"How would you assess this patient's {topic.replace('_', ' ')} function?",
                    ),
                    reference_answer=data.get("reference_answer", ""),
                    difficulty=difficulty,
                    ot_context=data.get("ot_context"),
                )
        except Exception as e:
            logger.error("Scenario generation failed: {e}", e=str(e))

        # Fallback
        return self._fallback_scenario(topic, template, difficulty)

    def _fallback_scenario(
        self,
        topic: str,
        template: str,
        difficulty: str,
    ) -> ClinicalScenario:
        topic_clean = topic.replace("_", " ")
        return ClinicalScenario(
            topic=topic,
            scenario_text=template,
            question=(
                f"Based on this presentation, what {topic_clean} structures "
                f"are involved and what are the OT implications?"
            ),
            reference_answer=(
                f"The {topic_clean} structures involved would cause specific "
                f"functional deficits. OT intervention would focus on compensatory "
                f"strategies, adaptive equipment, and functional retraining."
            ),
            difficulty=difficulty,
            ot_context=(
                f"Focus on functional implications of {topic_clean} "
                f"involvement for ADL performance."
            ),
        )
