"""
src/app/views/settings_page.py

Settings page — student profile, tutoring preferences, session management,
and a compact read-only system status. Preferences persist in session state
and are honoured live by the chat view.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

from src.config.settings import get_settings

NAVY = "#E2E8F0"
MUTED = "#94A3B8"

OT_LEVELS = [
    "Not specified",
    "Pre-OT / Prerequisite",
    "Year 1 (MSOT/OTD)",
    "Year 2 (MSOT/OTD)",
    "Year 3 (OTD)",
    "Fieldwork / Clinical",
    "Practising clinician",
]


def _section_title(text: str) -> None:
    st.markdown(
        f"<div style='font-size:14px;font-weight:600;color:{NAVY};margin-bottom:8px'>{text}</div>",
        unsafe_allow_html=True,
    )


def render() -> None:
    settings = get_settings()

    st.markdown(
        f"<h2 style='font-size:22px;font-weight:600;color:{NAVY};"
        f"margin-bottom:4px'>Settings</h2>"
        f"<p style='color:{MUTED};margin-bottom:0'>"
        f"Personalise your tutoring experience</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    col_left, col_right = st.columns(2, gap="large")

    # ── Profile ─────────────────────────────────────────────────────────────────
    with col_left:
        with st.container(border=True):
            _section_title("👤  Profile")
            name = st.text_input(
                "Your name",
                value=st.session_state.get("student_name", ""),
                placeholder="e.g. Alex Morgan",
            )
            current_level = st.session_state.get("student_level", "") or OT_LEVELS[0]
            level = st.selectbox(
                "OT program level",
                OT_LEVELS,
                index=OT_LEVELS.index(current_level) if current_level in OT_LEVELS else 0,
            )
            if st.button("Save profile", type="primary"):
                st.session_state.student_name = name.strip()
                st.session_state.student_level = "" if level == OT_LEVELS[0] else level
                # A named student gets a stable ID → cross-session memory (req 6).
                if name.strip():
                    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
                    st.session_state.student_id = f"student_{slug}" or st.session_state.student_id
                st.toast("Profile saved", icon="✅")

    # ── Tutoring preferences ─────────────────────────────────────────────────────
    with col_right:
        with st.container(border=True):
            _section_title("🎛  Tutoring preferences")
            st.toggle(
                "Show source citations",
                value=st.session_state.get("show_citations", True),
                key="show_citations",
                help="Display the textbook sources used to ground each answer.",
            )
            st.toggle(
                "Show hint progress",
                value=st.session_state.get("show_hints", True),
                key="show_hints",
                help="Show the hint-level indicator above the chat input.",
            )
            st.caption(
                f"Socratic strict mode is **{'on' if settings.socratic_strict_mode else 'off'}** "
                f"and the tutor reveals answers after **{settings.max_hint_turns} hints** "
                f"(configured in `.env`)."
            )

    # ── Accessibility ────────────────────────────────────────────────────────────
    with col_left:
        with st.container(border=True):
            _section_title("♿  Accessibility")
            st.toggle(
                "Read responses aloud (text-to-speech)",
                value=st.session_state.get("enable_tts", False),
                key="enable_tts",
                help="Adds a 🔊 button to tutor messages (OpenAI gpt-4o-mini-tts).",
            )
            st.toggle(
                "Voice answers (speech-to-text)",
                value=st.session_state.get("enable_stt", False),
                key="enable_stt",
                help="Upload an audio clip in chat; it's transcribed and sent "
                "(OpenAI gpt-4o-mini-transcribe).",
            )
            if (st.session_state.get("enable_tts") or st.session_state.get("enable_stt")) and (
                not settings.openai_api_key
            ):
                st.warning("Audio features require OPENAI_API_KEY in .env.", icon="⚠️")

    # ── Session management ───────────────────────────────────────────────────────
    with col_right:
        with st.container(border=True):
            _section_title("🗂  Session")
            st.caption(
                f"Phase **{st.session_state.get('phase', 'rapport').title()}** · "
                f"{st.session_state.get('turn_count', 0)} turns · "
                f"{len(st.session_state.get('topics_covered', []))} topics"
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Clear chat", width="stretch"):
                    st.session_state.messages = []
                    st.toast("Chat cleared", icon="🧹")
                    st.rerun()
            with cc2:
                if st.button("Reset session", width="stretch"):
                    for key in [
                        "messages",
                        "phase",
                        "turn_count",
                        "hint_level",
                        "current_topic",
                        "topics_covered",
                        "mastery_scores",
                    ]:
                        st.session_state.pop(key, None)
                    st.toast("Session reset", icon="🔄")
                    st.rerun()
            st.caption(
                "Reset session clears the current conversation only — your long-term "
                "mastery history is kept across sessions (so the tutor can revisit weak "
                "topics). To erase that too:"
            )
            if st.button("🗑  Clear learning history", width="stretch"):
                import asyncio

                from src.core.memory.student_memory import StudentMemory

                sid = st.session_state.get("student_id", "student_demo")
                try:
                    asyncio.run(StudentMemory().delete(sid))
                    st.session_state.pop("mastery_scores", None)
                    st.session_state.pop("topics_covered", None)
                    st.toast("Learning history cleared", icon="🗑")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not clear history: {e}")

    # ── System status (read-only, collapsed) ─────────────────────────────────────
    st.write("")
    with st.expander("System status", expanded=False):
        sc1, sc2 = st.columns(2)
        with sc1:
            llm = settings.openai_llm_model
            vision = settings.openai_vision_model
            key_ok = bool(settings.openai_api_key)
            st.code(
                f"Provider:   {settings.llm_provider.value}\n"
                f"LLM:        {llm}\n"
                f"Vision:     {vision}\n"
                f"API key:    {'set' if key_ok else 'MISSING'}"
            )
        with sc2:
            st.code(
                f"Vector store: {settings.vector_store_type.value}\n"
                f"Top-K:        {settings.top_k_retrieval}\n"
                f"Chunk size:   {settings.chunk_size}\n"
                f"Environment:  {settings.app_env.value}"
            )
        if settings.using_openai and not settings.openai_api_key:
            st.warning("OPENAI_API_KEY is not set in your .env file.", icon="⚠️")
