"""
src/core/audio/audio_service.py

AudioService — text-to-speech and speech-to-text for the accessibility layer.

Primary provider: OpenAI (gpt-4o-mini-tts / gpt-4o-mini-transcribe — the cheap
tier). Kept behind a small service class so a local/open-source backend
(pyttsx3, faster-whisper) can be dropped in later without touching the UI.
"""

from __future__ import annotations

import io

from src.config.settings import get_settings
from src.utils.logger import logger


class AudioError(Exception):
    """Raised when TTS/STT fails (missing key, API error, etc.)."""


class AudioService:
    """Thin wrapper over OpenAI audio endpoints."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = None

    def _get_client(self) -> object:
        if self._client is None:
            if not self._settings.openai_api_key:
                raise AudioError("OPENAI_API_KEY not set — audio features need it.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Text → spoken MP3 bytes. Raises AudioError on failure."""
        if not text or not text.strip():
            raise AudioError("Nothing to read aloud.")
        try:
            logger.info(
                "TTS synthesize: model={m} voice={v} chars={n}",
                m=self._settings.openai_tts_model,
                v=voice or self._settings.tts_voice,
                n=len(text[:4000]),
            )
            resp = self._get_client().audio.speech.create(
                model=self._settings.openai_tts_model,
                voice=voice or self._settings.tts_voice,
                input=text[:4000],  # TTS input cap — keep requests small/cheap
            )
            data = resp.content
            logger.info("TTS synthesize OK: {n} bytes", n=len(data))
            return data
        except AudioError:
            raise
        except Exception as e:
            logger.error("TTS failed: {e}", e=str(e))
            raise AudioError(f"Text-to-speech failed: {e}") from e

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Audio bytes → transcribed text. Raises AudioError on failure."""
        if not audio_bytes:
            raise AudioError("No audio to transcribe.")
        try:
            buf = io.BytesIO(audio_bytes)
            buf.name = filename  # OpenAI uses the extension to pick the decoder
            resp = self._get_client().audio.transcriptions.create(
                model=self._settings.openai_stt_model,
                file=buf,
            )
            return (resp.text or "").strip()
        except AudioError:
            raise
        except Exception as e:
            logger.error("STT failed: {e}", e=str(e))
            raise AudioError(f"Speech-to-text failed: {e}") from e
