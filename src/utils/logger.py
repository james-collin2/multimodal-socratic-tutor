"""
src/utils/logger.py

Centralised logging setup using loguru.
Import `logger` from here everywhere — never use stdlib logging directly.

Usage:
    from src.utils.logger import logger
    logger.info("RAG pipeline started")
    logger.debug("Retrieved {count} chunks", count=5)
    logger.error("LLM call failed: {error}", error=str(e))
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

from src.config.settings import get_settings


def _setup_logger() -> None:
    """Configure loguru with file + stderr handlers."""
    settings = get_settings()

    # Remove default handler
    _logger.remove()

    log_level = settings.app_log_level.value

    # ── stderr handler (coloured in dev, plain in prod) ───────────
    fmt_dev = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    fmt_prod = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}"

    _logger.add(
        sys.stderr,
        format=fmt_dev if settings.is_development else fmt_prod,
        level=log_level,
        colorize=settings.is_development,
        backtrace=settings.app_debug,
        diagnose=settings.app_debug,
    )

    # ── file handler (always plain JSON-like) ─────────────────────
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger.add(
        log_dir / "socratot_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        level=log_level,
        rotation=settings.log_rotation,
        compression="zip",
        retention="14 days",
        backtrace=True,
        diagnose=False,  # no sensitive data in file logs
    )


_setup_logger()

# Public export — import this everywhere
logger = _logger
