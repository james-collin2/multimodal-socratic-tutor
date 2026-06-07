"""
src/prompts/loader.py

Loads all prompt templates from config/prompts.yaml.
Provides type-safe access and string formatting for templates.

Usage:
    from src.prompts.loader import get_prompt
    prompt = get_prompt("socratic.hint_level_1", topic="cerebellum",
                        guiding_question="What role does balance play?")
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.exceptions import ConfigurationError
from src.utils.logger import logger

_PROMPTS_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"


@lru_cache(maxsize=1)
def _load_prompts() -> dict[str, Any]:
    """Load and cache the prompts.yaml file."""
    if not _PROMPTS_PATH.exists():
        raise ConfigurationError(
            f"prompts.yaml not found at {_PROMPTS_PATH}",
            detail="Ensure config/prompts.yaml exists in the project root",
        )
    with open(_PROMPTS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug("Loaded prompts from {path}", path=_PROMPTS_PATH)
    return data  # type: ignore[return-value]


def get_prompt(key: str, **kwargs: Any) -> str:
    """
    Retrieve and format a prompt template by dot-notation key.

    Args:
        key: Dot-separated path into prompts.yaml, e.g. "socratic.hint_level_1"
        **kwargs: Variables to substitute into the template

    Returns:
        Formatted prompt string

    Raises:
        ConfigurationError: If key is not found in prompts.yaml
    """
    prompts = _load_prompts()
    parts = key.split(".")

    node: Any = prompts
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise ConfigurationError(
                f"Prompt key not found: '{key}'",
                detail=(
                    f"Failed at segment '{part}'. Available keys: "
                    f"{list(node.keys()) if isinstance(node, dict) else '(not a dict)'}"
                ),
            )
        node = node[part]

    if not isinstance(node, str):
        raise ConfigurationError(
            f"Prompt key '{key}' does not resolve to a string",
            detail=f"Got type: {type(node).__name__}",
        )

    if kwargs:
        try:
            return node.format(**kwargs)
        except KeyError as e:
            raise ConfigurationError(
                f"Missing template variable in prompt '{key}': {e}",
                detail=f"Provided vars: {list(kwargs.keys())}",
            ) from e

    return node


def list_prompt_keys() -> list[str]:
    """Return all available prompt keys in dot-notation (for debugging)."""
    prompts = _load_prompts()

    def _flatten(d: dict[str, Any], prefix: str = "") -> list[str]:
        keys = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(_flatten(v, full_key))
            else:
                keys.append(full_key)
        return keys

    return _flatten(prompts)
