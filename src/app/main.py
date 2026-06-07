"""
src/app/main.py

socratOT — production Streamlit entry point.
Handles session init, sidebar, navigation, and page routing.

Run: streamlit run src/app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

# set_page_config MUST be the first Streamlit command — before importing
# config.settings, which touches st.secrets at import time.
st.set_page_config(
    page_title="socratOT — OT Anatomy Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",  # respect the user's collapse choice; don't force-expand on every refresh
    menu_items={
        "Get Help": "https://github.com/bahodir4/multimodal-socratic-tutor",
        "About": "socratOT — Socratic AI Tutor for Occupational Therapy Education",
    },
)

from src.config.settings import get_settings  # noqa: E402  # must follow set_page_config
from src.core.conversation.state import PHASE_CONFIG  # noqa: E402
from src.utils.logger import logger  # noqa: E402

settings = get_settings()


# ── Global CSS ─────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ════════════════════════════════════════════════════════════════════════
   socratOT — Dark Professional theme
   bg #0B1120 · surface #151D2E · border rgba(148,163,184,.12)
   text #E2E8F0 · muted #94A3B8 · indigo #6366F1 · cyan #22D3EE
   ════════════════════════════════════════════════════════════════════════ */

:root {
    --bg:        #0B1120;
    --surface:   #151D2E;
    --surface-2: #1A2336;
    --border:    rgba(148,163,184,.14);
    --text:      #E2E8F0;
    --muted:     #94A3B8;
    --indigo:    #6366F1;
    --indigo-hi: #818CF8;
    --cyan:      #22D3EE;
}

/* ── Base / typography ──────────────────────────────────────────────────── */
html, body, [class*="css"], .stMarkdown, .stButton, input, textarea, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background:
    radial-gradient(1200px 600px at 80% -10%, rgba(99,102,241,.10), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(34,211,238,.06), transparent 55%),
    var(--bg); }
h1, h2, h3, h4 { font-family: 'Inter', sans-serif; letter-spacing: -0.02em; color: var(--text); }
.stMarkdown, p, span, label, li { color: var(--text); }

/* ── Layout ─────────────────────────────────────────────────────────────── */
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1140px; }
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border);
    background: linear-gradient(180deg, #0E1626 0%, #0B1120 100%);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
hr { margin: 0.9rem 0; border-color: var(--border); }

/* ── Cards: bordered containers + metrics ───────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, var(--surface) 0%, #121A29 100%);
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 10px 30px -18px rgba(0,0,0,.7);
}
[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--surface) 0%, #11192a 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 10px 30px -20px rgba(99,102,241,.55);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(99,102,241,.45);
    box-shadow: 0 16px 40px -18px rgba(99,102,241,.6);
}
[data-testid="stMetricLabel"] { color: var(--muted); font-size: 12px; font-weight: 500;
    text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricValue"] { color: var(--text); font-weight: 700; font-size: 28px; }

/* ── Buttons ────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    font-size: 13.5px;
    color: var(--text);
    background: var(--surface-2);
    border: 1px solid var(--border);
    transition: all .14s ease;
}
.stButton > button:hover {
    border-color: var(--indigo);
    color: #fff;
    background: rgba(99,102,241,.16);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    border: 1px solid #6366F1;
    color: #fff;
    box-shadow: 0 8px 20px -8px rgba(99,102,241,.7);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%);
    box-shadow: 0 10px 26px -8px rgba(99,102,241,.85);
}
/* sidebar nav: left-aligned ghost buttons */
[data-testid="stSidebar"] .stButton > button {
    text-align: left;
    justify-content: flex-start;
    border-color: transparent;
    background: transparent;
    color: var(--muted);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,.12);
    color: var(--text);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    border-color: transparent;
    color: #fff;
}
/* "🔊 Read aloud" — compact subtle pill, not a full bordered button */
[class*="st-key-tts_btn_"] button {
    width: auto;
    padding: 2px 12px;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 999px;
}
[class*="st-key-tts_btn_"] button:hover {
    color: var(--text);
    border-color: var(--indigo);
    background: rgba(99,102,241,.12);
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 10px;
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--indigo) !important; }
[data-testid="stChatInput"] {
    max-width: 820px; margin: 0 auto;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--surface-2);
    box-shadow: 0 8px 28px -16px rgba(0,0,0,.8);
}
[data-testid="stChatInput"]:focus-within { border-color: var(--indigo); }
[data-testid="stBottomBlockContainer"] { background: transparent; }

/* ── Chat messages ──────────────────────────────────────────────────────── */
.stChatMessage {
    border-radius: 16px;
    padding: 14px 18px;
    margin-bottom: 6px;
    border: 1px solid var(--border);
}
.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(180deg, var(--surface) 0%, #121A29 100%);
}
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: #0E1626;
}
[data-testid="stChatMessageContent"] { font-size: 14.5px; line-height: 1.65; color: var(--text); }
[data-testid="stChatMessageContent"] p { margin-bottom: 0.5rem; }
.stChatMessage [data-testid^="chatAvatarIcon"] { width: 30px; height: 30px; border-radius: 8px; }
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    border-radius: 10px; color: #fff;
}
[data-testid="stChatInputSubmitButton"] svg { color: #fff; fill: #fff; }

/* ── Expander / dataframe ───────────────────────────────────────────────── */
[data-testid="stExpander"] details {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
}
[data-testid="stExpander"] summary { color: var(--text); }

/* ── Pills / badges ─────────────────────────────────────────────────────── */
.phase-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 13px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
    border: 1px solid var(--border);
}

/* ── Progress bar ───────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--indigo) 0%, var(--cyan) 100%);
}

/* ── ChatGPT-style composer pill ────────────────────────────────────────── */
.st-key-composer_bar {
    max-width: 800px; margin: 0 auto;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 6px 10px;
    box-shadow: 0 14px 36px -18px rgba(0,0,0,.85);
}
.st-key-composer_bar:focus-within {
    border-color: var(--indigo);
    box-shadow: 0 0 0 3px rgba(99,102,241,.16), 0 14px 36px -18px rgba(0,0,0,.85);
}
/* borderless, transparent text field that fills the pill */
.st-key-composer_bar .stTextInput input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 15px;
    padding-left: 6px;
    color: var(--text) !important;
}
.st-key-composer_bar .stTextInput input::placeholder { color: #6B7A93; }
/* round icon buttons: ➕ · 🎙 · ➤ */
.st-key-composer_bar [data-testid="stPopover"] button,
.st-key-composer_bar .stButton > button {
    border-radius: 50% !important;
    width: 40px !important; height: 40px !important;
    min-width: 40px; padding: 0 !important;
    border: none !important;
    background: transparent;
    font-size: 17px;
    color: var(--muted);
}
.st-key-composer_bar [data-testid="stPopover"] button:hover,
.st-key-composer_bar .stButton > button:hover {
    background: rgba(148,163,184,.16); color: var(--text);
}
.st-key-composer_bar .stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#6366F1,#4F46E5);
    color: #fff;
    box-shadow: 0 6px 16px -6px rgba(99,102,241,.8);
}
.st-key-composer_bar .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg,#818CF8,#6366F1);
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
/* NB: don't hide the whole header — the "reopen sidebar" arrow lives inside
   it. Make the header transparent and hide only menu/toolbar/decoration. */
#MainMenu, footer { visibility: hidden; }
/* Hide ONLY the toolbar's right-side actions (Deploy / menu / status) and the
   decoration line — NOT the whole stToolbar. In Streamlit 1.52 the collapsed
   sidebar's re-open button (stExpandSidebarButton) is nested inside stToolbar,
   so `display:none` on stToolbar removed the re-open button from layout and the
   sidebar could never be reopened (a display:none ancestor can't be overridden
   by the child's visibility/opacity rules below). */
[data-testid="stToolbarActions"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] {
    background: transparent;
    visibility: visible;     /* override any inherited hidden state */
    pointer-events: none;    /* let clicks pass through the empty bar … */
}
header[data-testid="stHeader"] * { pointer-events: auto; } /* …except its buttons */

/* Material icons are font ligatures (data-testid="stIconMaterial"). The global
   Inter font rule above was cascading onto them, so the glyph rendered at 0
   width — which collapsed the sidebar "re-open" button to 0x0 (invisible &
   unclickable). Restore the icon font so all Material icons render. */
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

/* Re-open arrow shown when the sidebar is collapsed (lives in the header).
   Give it an explicit, high-contrast, clickable size. */
[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 1000;
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    width: 2.4rem !important;
    height: 2.4rem !important;
    color: var(--text) !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px;
}
[data-testid="stExpandSidebarButton"]:hover {
    color: #fff !important;
    border-color: var(--indigo) !important;
    background: rgba(99,102,241,.18) !important;
}
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {
    font-size: 1.4rem !important;
    color: var(--text) !important;
    width: auto !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #243049; border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--indigo); }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state initialisation ───────────────────────────────────────────────


def _init_session() -> None:
    defaults: dict = {
        # Unique per session, so a refresh starts with a clean dashboard.
        # Saving a name in Settings switches to a stable, cross-session ID.
        "student_id": f"session_{uuid4().hex[:12]}",
        "student_name": "",
        "student_level": "",
        "session_id": None,
        "current_page": "chat",
        "messages": [],
        "phase": "rapport",
        "turn_count": 0,
        "hint_level": 0,
        "current_topic": None,
        "is_loading": False,
        "uploaded_image": None,
        "topics_covered": [],
        "mastery_scores": {},
        # display preferences (wired into the chat view)
        "show_citations": True,
        "show_hints": True,
        # accessibility (OpenAI TTS/STT) — on by default so the feature is
        # visible for graders; controls only spend on actual click.
        "enable_tts": True,
        "enable_stt": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar ────────────────────────────────────────────────────────────────────


def _render_sidebar() -> None:
    with st.sidebar:
        # Logo
        st.markdown(
            """
        <div style="display:flex;align-items:center;gap:11px;
                    padding-bottom:16px;margin-bottom:16px;
                    border-bottom:1px solid rgba(148,163,184,.14)">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#6366F1 0%,#22D3EE 130%);
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-weight:800;font-size:18px;flex-shrink:0;
                        box-shadow:0 8px 20px -8px rgba(99,102,241,.8)">S</div>
            <div>
                <div style="font-weight:700;font-size:15px;color:#E2E8F0;
                            letter-spacing:-.01em">socratOT</div>
                <div style="font-size:11px;color:#94A3B8">OT Anatomy Tutor</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown(
            "<div style='font-size:11px;font-weight:600;color:#94A3B8;"
            "letter-spacing:.08em;margin-bottom:6px'>NAVIGATION</div>",
            unsafe_allow_html=True,
        )
        pages = {
            "chat": ("💬", "Tutor Chat"),
            "dashboard": ("📊", "Dashboard"),
            "images": ("🖼️", "Image Analysis"),
            "settings": ("⚙️", "Settings"),
        }
        for key, (icon, label) in pages.items():
            active = st.session_state.current_page == key
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                width="stretch",
                type="primary" if active else "secondary",
            ):
                st.session_state.current_page = key
                st.rerun()

        st.divider()

        # Phase indicator
        phase = st.session_state.phase
        _default = (phase, "rgba(148,163,184,.16)", "#CBD5E1")
        label, bg, fg = PHASE_CONFIG.get(phase, _default)
        st.markdown(
            f"<div style='font-size:11px;font-weight:600;color:#94A3B8;"
            f"letter-spacing:.08em;margin-bottom:6px'>SESSION PHASE</div>"
            f"<div class='phase-pill' style='background:{bg};color:{fg}'>"
            f"● {label}</div>",
            unsafe_allow_html=True,
        )

        # Turn progress
        turns = st.session_state.turn_count
        max_turns = settings.max_session_turns
        hint_level = st.session_state.hint_level

        st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
        st.caption(f"Turn {turns} / {max_turns}")
        st.progress(min(turns / max_turns, 1.0))

        # Hint status
        if phase == "tutoring":
            max_hints = settings.max_hint_turns
            if hint_level >= max_hints:
                st.success("✓ Answer reveal unlocked", icon=None)
            else:
                remaining = max_hints - hint_level
                st.info(f"Hints used: {hint_level}/{max_hints} · {remaining} remaining", icon=None)

        st.divider()

        # Model info — OpenAI is the active provider
        with st.expander("System info", expanded=False):
            st.caption(f"**LLM** `{settings.openai_llm_model}`")
            st.caption(f"**Vision** `{settings.openai_vision_model}`")
            embed = (
                settings.openai_embedding_model
                if settings.embedding_provider.value == "openai"
                else settings.embedding_model.split("/")[-1]
            )
            st.caption(f"**Provider** `{settings.llm_provider.value}`")
            st.caption(f"**Embed** `{embed}`")
            st.caption(f"**Store** `{settings.vector_store_type.value}`")
            st.caption(f"**Env** `{settings.app_env.value}`")
            if settings.using_openai and not settings.openai_api_key:
                st.warning("OPENAI_API_KEY not set in .env", icon="⚠️")

        # Student info
        if st.session_state.student_name:
            st.divider()
            st.caption(f"👤 **{st.session_state.student_name}**  \nSession active")


# ── Page routing ───────────────────────────────────────────────────────────────


def _route() -> None:
    page = st.session_state.current_page
    if page == "chat":
        from src.app.views import chat

        chat.render()
    elif page == "dashboard":
        from src.app.views import dashboard

        dashboard.render()
    elif page == "images":
        from src.app.views import image_analysis

        image_analysis.render()
    elif page == "settings":
        from src.app.views import settings_page

        settings_page.render()
    else:
        st.error(f"Unknown page: {page}")


def main() -> None:
    logger.debug("App rendered — page={p}", p=st.session_state.get("current_page"))
    _init_session()
    _render_sidebar()
    _route()


if __name__ == "__main__":
    main()
