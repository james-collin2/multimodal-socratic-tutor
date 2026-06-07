"""
src/core/multimodal/vision_analyzer.py

VisionAnalyzer — sends anatomy images to GPT-4o vision.
Single responsibility: image in → identified structures + confidence out.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import get_settings
from src.utils.helpers import safe_parse_json
from src.utils.logger import logger


@dataclass
class VisionResult:
    """Result from one image analysis."""

    structures: list[str]  # identified anatomical structures
    region: str  # body region (brain, hand, etc.)
    confidence: float  # 0.0 – 1.0
    description: str  # free-text description
    ot_relevance: str  # OT clinical relevance
    raw_response: str  # full LLM response
    error: str | None = None


ANATOMY_VISION_PROMPT = """You are an expert anatomy educator for Occupational Therapy students.
Analyse this anatomical image carefully.

Respond ONLY with valid JSON — no preamble, no markdown fences:
{
  "structures": ["list", "of", "anatomical", "structures", "visible"],
  "region": "body region (e.g. brain, hand, upper_extremity, spinal_cord)",
  "confidence": 0.0-1.0,
  "description": "one paragraph describing what is shown",
  "ot_relevance": "one sentence on OT clinical relevance"
}

Be specific — name exact structures (e.g. 'median nerve' not just 'nerve').
Focus on structures relevant to Occupational Therapy practice."""


class VisionAnalyzer:
    """
    Analyses anatomy images using GPT-4o vision.

    Args:
        llm: Optional pre-built vision LLM (injected for testing)
    """

    def __init__(self, llm: object | None = None) -> None:
        self._llm = llm
        self._settings = get_settings()

    def _get_llm(self) -> object:
        if self._llm is None:
            from src.models.llm_factory import get_vision_llm

            self._llm = get_vision_llm()
        return self._llm

    async def analyze_file(self, image_path: Path) -> VisionResult:
        """Analyze an image file on disk."""
        if not image_path.exists():
            return VisionResult(
                structures=[],
                region="unknown",
                confidence=0.0,
                description="",
                ot_relevance="",
                raw_response="",
                error=f"File not found: {image_path}",
            )
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower().lstrip(".")
        media_type = "jpeg" if suffix in ("jpg", "jpeg") else suffix
        return await self.analyze_bytes(image_bytes, media_type)

    async def analyze_bytes(
        self,
        image_bytes: bytes,
        media_type: str = "jpeg",
    ) -> VisionResult:
        """Analyze raw image bytes — used for Streamlit uploaded files."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return await self._call_vision_llm(b64, media_type)

    async def _call_vision_llm(
        self,
        b64_image: str,
        media_type: str,
    ) -> VisionResult:
        """Send the image to the OpenAI vision model."""
        return await self._call_openai(b64_image, media_type)

    async def _call_openai(
        self,
        b64_image: str,
        media_type: str,
    ) -> VisionResult:
        """Call GPT-4o with base64 image."""
        try:
            from openai import OpenAI

            settings = get_settings()
            client = OpenAI(api_key=settings.openai_api_key)

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=settings.openai_vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{media_type};base64,{b64_image}",
                                        "detail": "high",
                                    },
                                },
                                {"type": "text", "text": ANATOMY_VISION_PROMPT},
                            ],
                        }
                    ],
                    max_tokens=600,
                ),
            )
            raw = response.choices[0].message.content or ""
            return self._parse_result(raw)

        except Exception as e:
            logger.error("GPT-4o vision failed: {e}", e=str(e))
            return VisionResult(
                structures=[],
                region="unknown",
                confidence=0.0,
                description="",
                ot_relevance="",
                raw_response="",
                error=str(e),
            )

    def _parse_result(self, raw: str) -> VisionResult:
        """Parse LLM JSON response into VisionResult."""
        parsed = safe_parse_json(raw)

        if not parsed:
            # Graceful fallback — extract what we can from free text
            logger.warning("Vision LLM returned non-JSON — using fallback parser")
            return VisionResult(
                structures=self._extract_structures_from_text(raw),
                region="anatomy",
                confidence=0.4,
                description=raw[:300] if raw else "Unable to parse response.",
                ot_relevance="Relevant to OT anatomy practice.",
                raw_response=raw,
            )

        return VisionResult(
            structures=parsed.get("structures", []),
            region=parsed.get("region", "anatomy"),
            confidence=float(parsed.get("confidence", 0.7)),
            description=parsed.get("description", ""),
            ot_relevance=parsed.get("ot_relevance", ""),
            raw_response=raw,
        )

    @staticmethod
    def _extract_structures_from_text(text: str) -> list[str]:
        """Extract structure names from free text as fallback."""
        anatomy_keywords = [
            "cerebellum",
            "cortex",
            "hippocampus",
            "thalamus",
            "brainstem",
            "median nerve",
            "ulnar nerve",
            "radial nerve",
            "brachial plexus",
            "spinal cord",
            "vertebra",
            "dermatome",
            "myotome",
            "carpal",
            "metacarpal",
            "phalanx",
            "tendon",
            "ligament",
            "rotator cuff",
            "deltoid",
            "trapezius",
            "biceps",
            "triceps",
        ]
        found = []
        text_lower = text.lower()
        for kw in anatomy_keywords:
            if kw in text_lower and kw not in found:
                found.append(kw)
        return found[:8]
