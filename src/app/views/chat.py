"""
src/app/views/chat.py

Production chat interface for socratOT.
Shows the Socratic tutoring conversation with:
  - Phase-aware top bar
  - Threaded chat with role-based bubbles
  - Source citation display
  - Hint level indicator
  - Image upload support
  - Bypass attempt feedback
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

from src.config.settings import get_settings
from src.core.conversation.state import PHASE_CONFIG
from src.utils.logger import logger

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

settings = get_settings()


def _phase_topbar() -> None:
    """Render the top status bar with phase, student, and model badges."""
    phase = st.session_state.get("phase", "rapport")
    name = st.session_state.get("student_name", "Guest")
    topic = st.session_state.get("current_topic", "")
    hint = st.session_state.get("hint_level", 0)
    max_h = settings.max_hint_turns

    label, bg, fg = PHASE_CONFIG.get(phase, ("Active", "rgba(148,163,184,.16)", "#CBD5E1"))

    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    with col1:
        st.markdown(
            f"<div style='font-size:15px;font-weight:700;color:#E2E8F0;letter-spacing:-.01em'>"
            f"Anatomy &amp; Neuroscience Tutor</div>"
            f"<div style='font-size:12px;color:#94A3B8'>"
            f"Student: {name}"
            f"{' · ' + topic.replace('_', ' ').title() if topic else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='margin-top:6px'>"
            f"<span style='background:{bg};color:{fg};padding:4px 12px;"
            f"border-radius:20px;font-size:12px;font-weight:600'>● {label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        rag_color = (
            "rgba(52,211,153,.16)" if settings.top_k_retrieval > 0 else "rgba(248,113,113,.16)"
        )
        rag_text = "#6EE7B7" if settings.top_k_retrieval > 0 else "#F87171"
        st.markdown(
            f"<div style='margin-top:6px'>"
            f"<span style='background:{rag_color};color:{rag_text};border:1px solid "
            f"rgba(148,163,184,.14);padding:4px 12px;border-radius:20px;font-size:12px;"
            f"font-weight:600'>● RAG active</span></div>",
            unsafe_allow_html=True,
        )
    with col4:
        if phase == "tutoring":
            if hint >= max_h:
                hint_color, hint_text = "rgba(52,211,153,.16)", "#6EE7B7"
                hint_label = "Reveal unlocked"
            else:
                hint_color, hint_text = "rgba(251,191,36,.16)", "#FBBF24"
                hint_label = f"Hint {hint}/{max_h}"
            st.markdown(
                f"<div style='margin-top:6px'>"
                f"<span style='background:{hint_color};color:{hint_text};"
                f"padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600'>"
                f"{hint_label}</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()


def _clean_for_speech(text: str) -> str:
    """Strip the citation block and markdown symbols so TTS reads naturally."""
    import re

    speech = re.split(r"\n\s*Sources?:", text)[0]  # drop trailing "Sources:" list
    speech = re.sub(r"[*_#`>\[\]]", "", speech)  # drop markdown punctuation
    speech = re.sub(r"\s+", " ", speech).strip()
    return speech or text


def _render_message(msg: dict, idx: int = 0) -> None:
    """Render a single chat message with role, content, and metadata."""
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    citations = msg.get("citations", [])
    hint_lvl = msg.get("hint_level")
    is_bypass = msg.get("is_bypass_redirect", False)
    is_reveal = msg.get("is_reveal", False)

    with st.chat_message(role, avatar="🧠" if role == "assistant" else "👤"):
        st.markdown(content)

        # Metadata row
        meta_parts = []
        if hint_lvl is not None and role == "assistant":
            hint_labels = {0: "Guiding question", 1: "Hint 1", 2: "Hint 2", 3: "Answer revealed"}
            hint_str = hint_labels.get(hint_lvl, f"Hint {hint_lvl}")
            if is_bypass:
                meta_parts.append(
                    "<span style='background:rgba(251,191,36,.16);color:#FBBF24;padding:2px 9px;"
                    "border-radius:10px;font-size:11px;font-weight:500'>⚠ Bypass redirected</span>"
                )
            elif is_reveal:
                meta_parts.append(
                    "<span style='background:rgba(52,211,153,.16);color:#6EE7B7;padding:2px 9px;"
                    "border-radius:10px;font-size:11px;font-weight:500'>✓ Answer revealed</span>"
                )
            else:
                meta_parts.append(
                    f"<span style='background:rgba(99,102,241,.18);color:#A5B4FC;padding:2px 9px;"
                    f"border-radius:10px;font-size:11px;font-weight:500'>"
                    f"💡 {hint_str}</span>"
                )

        if meta_parts:
            st.markdown(
                "<div style='margin-top:6px;display:flex;gap:6px;flex-wrap:wrap'>"
                + "".join(meta_parts)
                + "</div>",
                unsafe_allow_html=True,
            )

        # Citations
        if citations and st.session_state.get("show_citations", True):
            cite_html = " ".join(
                f"<span style='background:rgba(34,211,238,.12);color:#67E8F9;padding:2px 8px;"
                f"border:1px solid rgba(34,211,238,.22);border-radius:5px;font-size:11px;"
                f"font-family:monospace'>{c}</span>"
                for c in citations[:3]
            )
            st.markdown(
                f"<div style='margin-top:6px'>{cite_html}</div>",
                unsafe_allow_html=True,
            )

        # Read aloud (TTS) — accessibility
        if role == "assistant" and content and st.session_state.get("enable_tts"):
            audio_key = f"tts_audio_{idx}"
            if st.button("🔊 Read aloud", key=f"tts_btn_{idx}"):
                from src.core.audio.audio_service import AudioService

                try:
                    with st.spinner("Generating audio…"):
                        st.session_state[audio_key] = AudioService().synthesize(
                            _clean_for_speech(content)
                        )
                    st.session_state["_tts_autoplay"] = audio_key
                except Exception as e:  # surface ANY failure so it's never silent
                    st.session_state[audio_key] = None
                    st.error(f"Could not generate audio: {e}")
            if st.session_state.get(audio_key):
                # Auto-play only on the click that just generated it (not on every
                # later rerun), so old clips don't replay when the page refreshes.
                autoplay = st.session_state.get("_tts_autoplay") == audio_key
                if autoplay:
                    del st.session_state["_tts_autoplay"]
                _render_audio_player(st.session_state[audio_key], autoplay)


def _render_audio_player(data: bytes, autoplay: bool) -> None:
    """
    Play TTS audio via a self-contained base64 <audio> element.
    Avoids Streamlit's media-file server (which can error when audio bytes are
    re-served from session_state across reruns). Always shows controls so the
    student can press play if the browser blocks autoplay.
    """
    import base64

    import streamlit.components.v1 as components

    b64 = base64.b64encode(data).decode()
    auto = "autoplay" if autoplay else ""
    components.html(
        f'<audio {auto} controls style="width:100%;height:40px">'
        f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"></audio>',
        height=50,
    )


def _hint_indicator() -> None:
    """Show hint progress dots above the input."""
    phase = st.session_state.get("phase", "rapport")
    hint = st.session_state.get("hint_level", 0)
    max_hints = settings.max_hint_turns

    if phase != "tutoring" or not st.session_state.get("show_hints", True):
        return

    dots = ""
    for i in range(1, max_hints + 1):
        if i <= hint:
            dots += "<span style='color:#818CF8;font-size:14px'>●</span> "
        else:
            dots += "<span style='color:#33405C;font-size:14px'>●</span> "

    if hint >= max_hints:
        suffix = (
            "<span style='font-size:12px;color:#6EE7B7;font-weight:500'>"
            "Answer reveal unlocked</span>"
        )
    else:
        remaining = max_hints - hint
        suffix = (
            f"<span style='font-size:12px;color:#94A3B8'>"
            f"{remaining} hint{'s' if remaining > 1 else ''} remaining</span>"
        )

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
        f"<span style='font-size:12px;color:#94A3B8'>Hints:</span>"
        f"{dots}{suffix}</div>",
        unsafe_allow_html=True,
    )


def _is_substantive(text: str, engine: object) -> bool:
    """True if the message already names a topic or asks something teachable."""
    t = text.lower().strip()
    if engine.detect_topic(text):
        return True
    # Avoid bare "name"/"list" (they false-trigger on "my name is …").
    triggers = (
        "quiz",
        "explain",
        "what",
        "how",
        "why",
        "describe",
        "tell me about",
        "teach me",
        "list the",
        "name the",
        "where",
        "which",
        "function of",
        "?",
    )
    if "my name is" in t or t.startswith(("i am", "i'm", "hi", "hello", "hey")):
        return False
    return len(t.split()) >= 2 and any(k in t for k in triggers)


async def _get_tutor_response(user_input: str, uploaded_image: UploadedFile | None) -> dict:
    """
    Call the Socratic engine and return response dict.
    Runs async in event loop.
    """
    try:
        from src.core.conversation.socratic_engine import SocraticEngine
        from src.core.conversation.state import ConversationPhase

        if "_engine" not in st.session_state:
            st.session_state["_engine"] = SocraticEngine()
        engine = st.session_state["_engine"]

        # ── Image path: if an image is attached, analyse it (vision pipeline) ──────
        # The student's typed text (if any) is treated as a question about the image.
        if uploaded_image is not None:
            try:
                image_bytes = uploaded_image.getvalue()
            except AttributeError:
                image_bytes = uploaded_image.read()
            media_type = (getattr(uploaded_image, "type", "") or "image/jpeg").split("/")[-1]
            if media_type == "jpg":
                media_type = "jpeg"

            # The typed text (stored at send time) is the question about the image.
            question = (st.session_state.pop("image_question", "") or "").strip() or None
            img_response = await engine.generate_from_image(
                image_bytes=image_bytes,
                media_type=media_type,
                user_question=question,
            )
            # Move into tutoring so follow-up text turns escalate hints normally.
            st.session_state.phase = "tutoring"
            if img_response.topic_detected:
                st.session_state.current_topic = img_response.topic_detected
                if img_response.topic_detected not in st.session_state.topics_covered:
                    st.session_state.topics_covered.append(img_response.topic_detected)
            st.session_state.turn_count += 1
            return {
                "role": "assistant",
                "content": img_response.content,
                "citations": img_response.citations,
                "hint_level": None,
                "is_bypass_redirect": False,
                "is_reveal": False,
                "is_image_response": True,
            }

        phase = ConversationPhase(st.session_state.get("phase", "rapport"))

        # Skip rapport if the student's first message already states a topic or
        # asks something substantive — jump straight into tutoring so they don't
        # have to repeat themselves.
        if phase == ConversationPhase.RAPPORT and _is_substantive(user_input, engine):
            phase = ConversationPhase.TUTORING
            st.session_state.phase = "tutoring"

        state = {
            "turn_count": st.session_state.get("turn_count", 0),
            "hint_level": st.session_state.get("hint_level", 0),
            "student_name": st.session_state.get("student_name", ""),
            # prior turns (exclude the current user message) for context-aware retrieval
            "history": [
                {"role": m.get("role"), "content": m.get("content", "")}
                for m in st.session_state.messages[:-1][-6:]
            ],
        }

        response = await engine.generate(
            student_input=user_input,
            session_state=state,
            phase=phase,
        )

        # Update session state
        st.session_state.turn_count += 1
        st.session_state.hint_level = int(response.hint_level)
        if response.topic_detected:
            st.session_state.current_topic = response.topic_detected
            if response.topic_detected not in st.session_state.topics_covered:
                st.session_state.topics_covered.append(response.topic_detected)

        # Advance to tutoring after a single rapport exchange (the UI already
        # showed the opening greeting, so one acknowledgement is enough).
        if phase == ConversationPhase.RAPPORT and st.session_state.turn_count >= 1:
            st.session_state.phase = "tutoring"

        # ── Assessment + memory connection (feeds the dashboard) ───────────────
        # Score the student's contribution against the retrieved context and
        # persist a per-topic mastery score to StudentMemory + session state.
        # Wrapped so a failure never breaks the chat (blind-test safety).
        topic = response.topic_detected or st.session_state.get("current_topic")
        if (
            phase == ConversationPhase.TUTORING
            and topic
            and response.retrieved_context
            and user_input.strip()
        ):
            try:
                from src.core.conversation.evaluator import StudentResponseEvaluator
                from src.core.memory.student_memory import StudentMemory

                overlap = StudentResponseEvaluator().keyword_overlap(
                    user_input, response.retrieved_context
                )
                new_score = round(overlap * 100, 1)
                prev = st.session_state.mastery_scores.get(topic)
                blended = new_score if prev is None else round(0.6 * prev + 0.4 * new_score, 1)
                st.session_state.mastery_scores[topic] = blended
                await StudentMemory().update_topic_score(
                    st.session_state.get("student_id", "student_demo"), topic, blended
                )
            except Exception as e:
                logger.warning("Mastery update skipped: {e}", e=str(e))

        return {
            "role": "assistant",
            "content": response.content,
            "citations": response.citations,
            "hint_level": int(response.hint_level),
            "is_bypass_redirect": response.is_bypass_redirect,
            "is_reveal": response.is_reveal,
        }

    except Exception as e:
        logger.error("Tutor response error: {e}", e=str(e))
        return {
            "role": "assistant",
            "content": (
                "I encountered an error generating a response. "
                "Please check your OpenAI API key in settings and try again."
            ),
            "citations": [],
            "hint_level": st.session_state.get("hint_level", 0),
            "is_bypass_redirect": False,
            "is_reveal": False,
        }


STARTER_PROMPTS = [
    "What does the cerebellum do?",
    "Walk me through the median nerve pathway",
    "Quiz me on the cranial nerves",
    "Why do dermatomes matter in OT?",
]

WELCOME = (
    "Welcome! I am **socratOT**, your Socratic anatomy and neuroscience tutor. "
    "I teach through guided questioning — I will never just hand you the answer, "
    "but I will guide you to discover it yourself.\n\n"
    "Before we begin, may I ask your name and which semester of your OT programme "
    "you are currently in?"
)


def _send(text: str) -> None:
    """Append a user message and trigger the tutor response on the next rerun."""
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.is_loading = True
    st.rerun()


def _composer() -> None:
    """
    ChatGPT-style bottom composer: ➕ attach · editable text · 🎙 mic · ➤ send.
    The mic records live, transcribes (OpenAI STT) and drops the text into the
    editable box so the student can review/edit before sending — never auto-sends.
    """
    # Clear the box on the run *after* a send (can't mutate a widget's state
    # in the same run it was created, so we defer via a flag).
    if st.session_state.pop("_clear_composer", False):
        st.session_state.composer_text = ""
        st.session_state.pop("_last_audio_hash", None)

    st.session_state.setdefault("composer_text", "")
    # Rotating key so we can RESET the file_uploader after an image is consumed —
    # otherwise the widget keeps handing back the same file and every later text
    # turn would be re-analysed as an image.
    st.session_state.setdefault("uploader_seq", 0)

    disabled = st.session_state.is_loading
    attached = st.session_state.get("uploaded_image")

    # One horizontal container styled (via .st-key-composer_bar) into a single
    # ChatGPT-style pill: ➕ attach · text (grows) · 🎙 mic · ➤ send.
    with st.container(
        key="composer_bar", horizontal=True, vertical_alignment="center", gap="small"
    ):
        # ── ➕ Attach image ────────────────────────────────────────────────────
        with st.popover("📎" if attached else "➕", width="content"):
            uploaded = st.file_uploader(
                "Attach an anatomy image",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key=f"composer_img_{st.session_state.uploader_seq}",
            )
            if uploaded is not None:
                st.session_state.uploaded_image = uploaded
                st.image(uploaded, width=180, caption="Attached — add a question & send")
            if attached and st.button("Remove", key="composer_img_rm"):
                st.session_state.uploaded_image = None
                st.session_state.uploader_seq += 1  # reset the uploader widget
                st.rerun()

        # ── 🎙 Mic — created BEFORE the text box so transcription can fill it ───
        audio = None
        with st.popover("🎙", width="content"):
            if hasattr(st, "audio_input"):
                st.caption("Record, then it transcribes into the message box.")
                audio = st.audio_input(
                    "Record", label_visibility="collapsed", key="composer_mic", disabled=disabled
                )
            else:
                st.caption("Voice input needs Streamlit ≥ 1.43.")
        if audio is not None:
            try:
                data = audio.getvalue()
            except Exception:  # defensive: treat unreadable audio as none
                data = b""
            if data and hash(data) != st.session_state.get("_last_audio_hash"):
                st.session_state._last_audio_hash = hash(data)
                from src.core.audio.audio_service import AudioError, AudioService

                try:
                    with st.spinner("Transcribing…"):
                        text = AudioService().transcribe(data, filename="mic.wav")
                    if text:
                        prev = st.session_state.get("composer_text", "").strip()
                        st.session_state.composer_text = f"{prev} {text}".strip() if prev else text
                        st.rerun()
                    else:
                        st.toast("No speech detected", icon="🎙")
                except AudioError as e:
                    st.error(str(e))

        # ── Editable text box (grows to fill the pill) ─────────────────────────
        st.text_input(
            "Message",
            key="composer_text",
            placeholder="Message socratOT…",
            label_visibility="collapsed",
            disabled=disabled,
            width="stretch",
        )

        # ── ➤ Send ─────────────────────────────────────────────────────────────
        send = st.button("➤", width="content", type="primary", disabled=disabled)

    if attached:
        st.caption(f"📎 {getattr(attached, 'name', 'image')} attached — analysed on send")

    if send:
        text = st.session_state.get("composer_text", "").strip()
        img = st.session_state.get("uploaded_image")
        if text or img is not None:
            # Remember the typed text as the image question (image path reads this).
            st.session_state.image_question = text
            display = text if text else "📷 Image uploaded for analysis"
            st.session_state.messages.append({"role": "user", "content": display})
            st.session_state.is_loading = True
            st.session_state._clear_composer = True
            st.rerun()


def _starter_suggestions() -> None:
    """ChatGPT-style starter prompts, shown until the student's first message."""
    st.markdown(
        "<div style='text-align:center;color:#94A3B8;font-size:13px;"
        "margin:18px 0 10px'>Not sure where to start? Try one of these</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, prompt in enumerate(STARTER_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"starter_{i}", width="stretch"):
                _send(prompt)


def render() -> None:
    """Main chat page render function."""

    _phase_topbar()

    # Seed the opening greeting once
    if not st.session_state.messages:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": WELCOME,
                "citations": [],
                "hint_level": None,
                "is_bypass_redirect": False,
                "is_reveal": False,
            }
        )

    # ── Render chat history ────────────────────────────────────────────────────
    for idx, msg in enumerate(st.session_state.messages):
        _render_message(msg, idx)

    # ── Loading indicator ─────────────────────────────────────────────────────
    if st.session_state.is_loading:
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Thinking..."):
                st.empty()

    # ── Starter prompts (only before the first user message) ────────────────────
    has_user_msg = any(m.get("role") == "user" for m in st.session_state.messages)
    if not has_user_msg and not st.session_state.is_loading:
        _starter_suggestions()

    # ── Hint indicator ─────────────────────────────────────────────────────────
    _hint_indicator()

    # ── Composer (attach · text · mic · send) ──────────────────────────────────
    _composer()

    # ── Process response after rerun ──────────────────────────────────────────
    if (
        st.session_state.is_loading
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
    ):
        last_input = st.session_state.messages[-1]["content"]
        with st.spinner("socratOT is thinking..."):
            response = asyncio.run(
                _get_tutor_response(
                    last_input,
                    st.session_state.uploaded_image,
                )
            )

        st.session_state.messages.append(response)
        st.session_state.is_loading = False
        # Image consumed — clear it AND reset the uploader widget so the next
        # text turn isn't re-analysed as an image.
        if st.session_state.get("uploaded_image") is not None:
            st.session_state.uploaded_image = None
            st.session_state.uploader_seq = st.session_state.get("uploader_seq", 0) + 1
        st.rerun()
