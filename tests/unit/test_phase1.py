"""
tests/unit/test_phase1.py

Phase 1 milestone verification tests.
All tests must pass before moving to Phase 2.

Run:  pytest tests/unit/test_phase1.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ── Settings tests ────────────────────────────────────────────────────────────


class TestSettings:
    def test_settings_load_with_defaults(self) -> None:
        """Settings resolves built-in defaults when no .env is loaded."""
        from src.config.settings import Settings

        # _env_file=None ignores any local .env so we test the real defaults
        # (otherwise a developer's .env override would make this flaky).
        s = Settings(_env_file=None)
        assert s.openai_llm_model == "gpt-4o-mini"
        assert s.chunk_size == 512
        assert s.chunk_overlap == 64
        assert s.top_k_retrieval == 5

    def test_settings_singleton(self) -> None:
        """get_settings() returns the same instance each call."""
        from src.config.settings import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_computed_properties(self) -> None:
        """Computed properties resolve correctly."""
        from src.config.settings import AppEnv, Settings

        s = Settings(app_env=AppEnv.DEVELOPMENT)
        assert s.is_development is True
        assert s.is_production is False

    def test_chunk_overlap_validation(self) -> None:
        """Bad config is clamped (not crashed): overlap is forced below chunk_size."""
        from src.config.settings import Settings

        s = Settings(chunk_size=512, chunk_overlap=9999)
        assert s.chunk_overlap < s.chunk_size

    def test_out_of_range_settings_clamped(self) -> None:
        """Out-of-range numeric settings clamp into range instead of raising."""
        from src.config.settings import Settings

        s = Settings(max_hint_turns=0, top_k_retrieval=999)
        assert s.max_hint_turns == 1  # clamped up to minimum
        assert s.top_k_retrieval == 20  # clamped down to maximum


# ── Schema tests ──────────────────────────────────────────────────────────────


class TestConversationSchemas:
    def test_session_state_defaults(self) -> None:
        from src.schemas.conversation import SessionState

        state = SessionState(student_id="student_001")
        assert state.turn_count == 0
        assert state.hint_turns_used == 0
        assert state.is_active is True

    def test_answer_reveal_eligibility(self) -> None:
        from src.schemas.conversation import SessionState

        state = SessionState(student_id="student_001", hint_turns_used=2)
        assert state.answer_reveal_eligible is True

    def test_conversation_history_turn_count(self) -> None:
        from src.schemas.conversation import (
            ConversationHistory,
            Message,
            MessageRole,
        )

        history = ConversationHistory()
        history.add(Message(role=MessageRole.USER, content="Hello", turn_number=1))
        history.add(Message(role=MessageRole.ASSISTANT, content="Hi!", turn_number=1))
        history.add(Message(role=MessageRole.USER, content="Question", turn_number=2))
        assert history.turn_count == 2

    def test_session_state_phase_display(self) -> None:
        from src.schemas.conversation import ConversationPhase, SessionState

        state = SessionState(student_id="s1", phase=ConversationPhase.TUTORING)
        assert state.phase_display == "Tutoring"


class TestAssessmentSchemas:
    def test_topic_mastery_update(self) -> None:
        from src.schemas.assessment import MasteryLevel, TopicMastery

        t = TopicMastery(topic="cerebellum")
        assert t.score == 0.0
        t.update(90.0)
        assert t.attempts == 1
        assert t.score > 0.0
        assert t.mastery_level == MasteryLevel.NOVICE  # first update from 0

    def test_reasoning_score_passed(self) -> None:
        from src.schemas.assessment import MasteryLevel, ReasoningScore

        score = ReasoningScore(
            clinical_accuracy=35,
            reasoning_quality=30,
            terminology=15,
            total=80,
            feedback="Good work",
            mastery_level=MasteryLevel.PROFICIENT,
        )
        assert score.passed is True

    def test_reasoning_score_failed(self) -> None:
        from src.schemas.assessment import MasteryLevel, ReasoningScore

        score = ReasoningScore(
            clinical_accuracy=10,
            reasoning_quality=10,
            terminology=5,
            total=25,
            feedback="Needs work",
            mastery_level=MasteryLevel.NOVICE,
        )
        assert score.passed is False


class TestRAGSchemas:
    def test_retrieval_result_has_results(self) -> None:
        from src.schemas.rag import RetrievalResult

        result = RetrievalResult(query="cerebellum function")
        assert result.has_results is False
        assert result.top_score == 0.0


# ── Utility tests ─────────────────────────────────────────────────────────────


class TestHelpers:
    def test_truncate_text(self) -> None:
        from src.utils.helpers import truncate_text

        long = "a" * 600
        result = truncate_text(long, max_chars=100)
        assert len(result) == 100
        assert result.endswith("...")

    def test_truncate_text_short(self) -> None:
        from src.utils.helpers import truncate_text

        short = "hello"
        assert truncate_text(short, max_chars=100) == "hello"

    def test_clean_text(self) -> None:
        from src.utils.helpers import clean_text

        messy = "  Hello   \n\n  World  "
        assert clean_text(messy) == "Hello World"

    def test_safe_parse_json_valid(self) -> None:
        from src.utils.helpers import safe_parse_json

        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_safe_parse_json_with_fences(self) -> None:
        from src.utils.helpers import safe_parse_json

        result = safe_parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_safe_parse_json_invalid(self) -> None:
        from src.utils.helpers import safe_parse_json

        assert safe_parse_json("not json at all") is None

    def test_detect_bypass_attempt(self) -> None:
        from src.utils.helpers import detect_bypass_attempt

        assert detect_bypass_attempt("just tell me the answer") is True
        assert detect_bypass_attempt("give me the answer please") is True
        assert detect_bypass_attempt("I think it relates to balance") is False

    def test_hash_string_consistent(self) -> None:
        from src.utils.helpers import hash_string

        assert hash_string("hello") == hash_string("hello")
        assert hash_string("hello") != hash_string("world")

    def test_chunk_list(self) -> None:
        from src.utils.helpers import chunk_list

        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_format_citations(self) -> None:
        from src.utils.helpers import format_citations

        sources = ["OpenStax ch14", "MedPix img01", "OpenStax ch14"]
        result = format_citations(sources)
        assert "[1]" in result
        assert "[2]" in result
        # Deduplicated — should only have 2 entries
        assert "[3]" not in result


# ── Prompt loader tests ───────────────────────────────────────────────────────


class TestPromptLoader:
    def test_load_known_key(self) -> None:
        from src.prompts.loader import get_prompt

        prompt = get_prompt("socratic.rapport_opener")
        assert len(prompt) > 10
        assert "socratOT" in prompt

    def test_load_with_format_vars(self) -> None:
        from src.prompts.loader import get_prompt

        prompt = get_prompt(
            "socratic.hint_level_1",
            topic="cerebellum",
            guiding_question="What role does balance play?",
        )
        assert "cerebellum" in prompt
        assert "balance" in prompt

    def test_unknown_key_raises(self) -> None:
        from src.prompts.loader import get_prompt
        from src.utils.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            get_prompt("nonexistent.key")

    def test_list_keys_not_empty(self) -> None:
        from src.prompts.loader import list_prompt_keys

        keys = list_prompt_keys()
        assert len(keys) > 0
        assert "socratic.rapport_opener" in keys


# ── Project structure tests ───────────────────────────────────────────────────


class TestProjectStructure:
    ROOT = Path(__file__).parent.parent.parent

    def test_required_files_exist(self) -> None:
        required = [
            ".env.example",
            ".gitignore",
            "requirements.txt",
            "pyproject.toml",
            "Dockerfile",
            "docker-compose.yml",
            "src/config/settings.py",
            "src/config/prompts.yaml",
            "src/config/topics.yaml",
            "src/schemas/base.py",
            "src/schemas/conversation.py",
            "src/schemas/rag.py",
            "src/schemas/assessment.py",
            "src/utils/logger.py",
            "src/utils/exceptions.py",
            "src/utils/helpers.py",
            "src/models/llm_factory.py",
            "src/models/embedding_model.py",
            "src/prompts/loader.py",
            "scripts/setup.py",
        ]
        for f in required:
            path = self.ROOT / f
            assert path.exists(), f"Missing required file: {f}"

    def test_required_dirs_exist(self) -> None:
        required_dirs = [
            "src/core/conversation",
            "src/core/rag",
            "src/core/multimodal",
            "src/core/memory",
            "src/core/assessment",
            "src/models",
            "src/schemas",
            "src/utils",
            "src/prompts",
            "data/raw",
            "data/processed",
            "src/config",
            "tests/unit",
            "tests/integration",
        ]
        for d in required_dirs:
            path = self.ROOT / d
            assert path.is_dir(), f"Missing required directory: {d}"

    def test_env_example_has_required_keys(self) -> None:
        env_example = (self.ROOT / ".env.example").read_text()
        required_keys = [
            "OPENAI_LLM_MODEL",
            "OPENAI_VISION_MODEL",
            "EMBEDDING_MODEL",
            "CHUNK_SIZE",
            "DATABASE_URL",
        ]
        for key in required_keys:
            assert key in env_example, f"Missing env key: {key}"
