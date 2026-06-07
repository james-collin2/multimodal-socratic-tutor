"""
src/core/multimodal/question_generator.py

ImageQuestionGenerator — generates Socratic questions from identified structures.
Single responsibility: VisionResult in → list of Socratic questions out.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.core.multimodal.vision_analyzer import VisionResult
from src.utils.helpers import safe_parse_json
from src.utils.logger import logger


@dataclass
class ImageQuestion:
    """One Socratic question generated from an anatomy image."""

    question: str
    structure: str  # which structure this targets
    difficulty: str  # beginner | intermediate | advanced
    hint: str  # hint to give if student struggles


class ImageQuestionGenerator:
    """
    Generates Socratic questions from VisionResult structures.

    Args:
        llm: Optional LLM instance (injected for testing)
    """

    def __init__(self, llm: object | None = None) -> None:
        self._llm = llm

    def _get_llm(self) -> object:
        if self._llm is None:
            from src.models.llm_factory import get_llm

            self._llm = get_llm()
        return self._llm

    async def generate(
        self,
        vision_result: VisionResult,
        n_questions: int = 3,
    ) -> list[ImageQuestion]:
        """
        Generate n Socratic questions from identified structures.

        Args:
            vision_result: Output from VisionAnalyzer
            n_questions:   How many questions to generate (default 3)

        Returns:
            List of ImageQuestion objects, ordered easy → hard
        """
        if not vision_result.structures:
            return self._fallback_questions(vision_result.region)

        structures_str = ", ".join(vision_result.structures[:6])

        prompt = f"""You are a Socratic OT anatomy tutor. Given these identified structures
from an anatomical image, generate {n_questions} Socratic questions for OT students.

Identified structures: {structures_str}
Body region: {vision_result.region}
OT relevance: {vision_result.ot_relevance}

Rules:
- NEVER give the answer — only ask guiding questions
- Order questions: beginner → intermediate → advanced
- Each question should target a specific structure
- Include a short hint (not the answer) for each

Respond ONLY with valid JSON array:
[
  {{
    "question": "What do you think the role of [structure] is in...",
    "structure": "structure name",
    "difficulty": "beginner",
    "hint": "Think about what happens when this structure is damaged..."
  }}
]"""

        try:
            llm = self._get_llm()
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: llm.invoke(prompt),
            )
            raw = resp.content if hasattr(resp, "content") else str(resp)
            parsed = safe_parse_json(raw)

            if isinstance(parsed, list):
                questions = []
                for item in parsed[:n_questions]:
                    questions.append(
                        ImageQuestion(
                            question=item.get("question", ""),
                            structure=item.get("structure", ""),
                            difficulty=item.get("difficulty", "intermediate"),
                            hint=item.get("hint", ""),
                        )
                    )
                logger.info(
                    "Generated {n} questions for region={r}",
                    n=len(questions),
                    r=vision_result.region,
                )
                return questions

        except Exception as e:
            logger.error("Question generation failed: {e}", e=str(e))

        return self._fallback_questions(vision_result.region)

    def _fallback_questions(self, region: str) -> list[ImageQuestion]:
        """Return generic Socratic questions when LLM fails."""
        fallbacks = {
            "brain": [
                ImageQuestion(
                    question=(
                        "What function do you think this brain region serves based on its location?"
                    ),
                    structure="brain region",
                    difficulty="beginner",
                    hint="Consider what deficits patients show when this area is damaged.",
                ),
                ImageQuestion(
                    question=(
                        "How might damage to this area affect an OT patient's "
                        "ability to perform ADLs?"
                    ),
                    structure="brain region",
                    difficulty="intermediate",
                    hint="Think about the motor or cognitive functions this region controls.",
                ),
            ],
            "hand": [
                ImageQuestion(
                    question=(
                        "Which structures visible here are most critical for precision pinch grip?"
                    ),
                    structure="hand",
                    difficulty="beginner",
                    hint="Consider the thenar muscles and their innervation.",
                ),
                ImageQuestion(
                    question=(
                        "If a patient had a laceration here, which functional "
                        "movements would be most affected?"
                    ),
                    structure="hand",
                    difficulty="intermediate",
                    hint="Think about which tendons and nerves pass through this region.",
                ),
            ],
        }
        defaults = [
            ImageQuestion(
                question=(
                    "What anatomical structures can you identify in this image "
                    "and why are they clinically important?"
                ),
                structure="anatomy",
                difficulty="beginner",
                hint="Start by identifying the major landmarks first.",
            ),
            ImageQuestion(
                question=(
                    "How might a lesion or injury to the structures shown here "
                    "present clinically in an OT patient?"
                ),
                structure="anatomy",
                difficulty="intermediate",
                hint="Consider both motor and sensory consequences.",
            ),
        ]
        return fallbacks.get(region, defaults)
