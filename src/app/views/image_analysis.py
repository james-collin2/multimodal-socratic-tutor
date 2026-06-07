"""
src/app/views/image_analysis.py

Image analysis page.
Accepts anatomy image uploads, runs GPT-4o vision, and generates
Socratic questions. Displays identified structures and local corpus.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

NAVY = "#E2E8F0"
MUTED = "#94A3B8"


def _section_title(text: str) -> None:
    st.markdown(
        f"<div style='font-size:14px;font-weight:600;color:{NAVY};margin-bottom:8px'>{text}</div>",
        unsafe_allow_html=True,
    )


def _structure_badge(structure: str) -> str:
    return (
        f"<span style='background:rgba(99,102,241,.16);color:#A5B4FC;padding:3px 11px;"
        f"border:1px solid rgba(99,102,241,.28);border-radius:12px;font-size:12px;"
        f"font-weight:500;margin:2px;display:inline-block'>{structure}</span>"
    )


async def _run_vision_pipeline(
    image_bytes: bytes,
    media_type: str,
    user_question: str | None = None,
) -> dict:
    """Run multimodal pipeline and return result dict."""
    try:
        from src.core.multimodal.pipeline import MultimodalPipeline

        pipeline = MultimodalPipeline()
        result = await pipeline.process_image(image_bytes, media_type, user_question=user_question)
        return {
            "ok": True,
            "reply": result.socratic_reply,
            "structures": result.vision_result.structures,
            "region": result.vision_result.region,
            "confidence": result.vision_result.confidence,
            "ot_relevance": result.vision_result.ot_relevance,
            "description": result.vision_result.description,
            "questions": result.questions,
            "citations": result.citations,
            "error": result.vision_result.error,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def render() -> None:
    """Main image analysis page."""
    st.markdown(
        f"<h2 style='font-size:22px;font-weight:600;color:{NAVY};margin-bottom:4px'>"
        f"Anatomical Image Analysis</h2>"
        f"<p style='color:{MUTED};margin-bottom:0'>"
        f"Upload any anatomy diagram — GPT-4o identifies structures "
        f"and generates Socratic questions</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        with st.container(border=True):
            _section_title("Upload image")
            uploaded = st.file_uploader(
                "Upload anatomy image",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                help="Any anatomy diagram — does not need to be from the training set.",
            )
            if uploaded:
                st.image(uploaded, caption=uploaded.name, width="stretch")
                st.caption(f"{uploaded.type} · {uploaded.size // 1024} KB")

                question = st.text_input(
                    "Ask a question about this image (optional)",
                    placeholder="e.g. Which nerve is highlighted and what does it innervate?",
                    key="vision_question",
                )

                if st.button("🔬 Analyse image", width="stretch", type="primary"):
                    with st.spinner("GPT-4o analysing structures..."):
                        image_bytes = uploaded.read()
                        media_type = uploaded.type.split("/")[-1]
                        if media_type == "jpg":
                            media_type = "jpeg"

                        result = asyncio.run(
                            _run_vision_pipeline(
                                image_bytes, media_type, (question or "").strip() or None
                            )
                        )

                    st.session_state["vision_result"] = result
                    st.session_state["vision_image_name"] = uploaded.name
                    st.rerun()

    with col2:
        result = st.session_state.get("vision_result")

        if result and result.get("ok"):
            # Record the detected region as an explored topic so image work
            # shows up on the dashboard (connects this page to the rest).
            region = result.get("region", "")
            if region:
                topics = st.session_state.setdefault("topics_covered", [])
                if region not in topics:
                    topics.append(region)

            # ── Structures identified ─────────────────────────────────────
            with st.container(border=True):
                _section_title("Identified structures")
                structures = result.get("structures", [])
                confidence = result.get("confidence", 0.0)

                if structures:
                    badges = " ".join(_structure_badge(s) for s in structures)
                    st.markdown(badges, unsafe_allow_html=True)
                    st.write("")
                    col_r, col_c = st.columns(2)
                    col_r.metric("Region", region.replace("_", " ").title())
                    col_c.metric("Confidence", f"{confidence * 100:.0f}%")

                    if result.get("ot_relevance"):
                        st.caption(f"**OT relevance:** {result['ot_relevance']}")
                else:
                    st.warning("No structures identified — try a clearer image.")

            # ── Socratic questions ────────────────────────────────────────
            questions = result.get("questions", [])
            if questions:
                with st.container(border=True):
                    _section_title("Socratic questions")
                    for i, q in enumerate(questions, 1):
                        diff_colors = {
                            "beginner": ("rgba(52,211,153,.14)", "#6EE7B7"),
                            "intermediate": ("rgba(251,191,36,.14)", "#FBBF24"),
                            "advanced": ("rgba(244,150,90,.14)", "#F6B07D"),
                        }
                        bg, fg = diff_colors.get(
                            getattr(q, "difficulty", "intermediate"),
                            ("rgba(148,163,184,.14)", "#CBD5E1"),
                        )
                        diff_label = getattr(q, "difficulty", "intermediate").title()
                        question_text = getattr(q, "question", str(q))

                        st.markdown(
                            f"<div style='background:{bg};border-radius:8px;"
                            f"padding:10px 14px;margin-bottom:8px'>"
                            f"<div style='font-size:11px;color:{fg};font-weight:600;"
                            f"margin-bottom:4px'>Q{i} · {diff_label}</div>"
                            f"<div style='font-size:13px;color:{NAVY}'>{question_text}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Send to chat
                    if st.button("💬 Continue in tutor chat", width="stretch"):
                        first_q = getattr(questions[0], "question", "") if questions else ""
                        if first_q:
                            if "messages" not in st.session_state:
                                st.session_state.messages = []
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": result.get("reply", first_q),
                                    "citations": result.get("citations", []),
                                    "hint_level": 0,
                                    "is_bypass_redirect": False,
                                    "is_reveal": False,
                                    "is_image_response": True,
                                }
                            )
                        st.session_state.current_page = "chat"
                        st.rerun()

            # ── Citations ─────────────────────────────────────────────────
            if result.get("citations"):
                with st.container(border=True):
                    _section_title("Sources")
                    for c in result["citations"]:
                        st.caption(f"📄 {c}")

        elif result and not result.get("ok"):
            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            st.info("Ensure your OpenAI API key is set and try again.")

        else:
            # No result yet
            with st.container(border=True):
                _section_title("How it works")
                st.info(
                    "Upload any anatomical diagram and click **Analyse image**. "
                    "GPT-4o vision will identify structures and generate "
                    "3 Socratic questions ordered by difficulty.",
                    icon="🔬",
                )
                st.markdown(
                    f"<div style='font-size:13px;color:{MUTED};margin-top:8px'>"
                    f"Supported: brain · upper extremity · hand · "
                    f"spinal cord · nervous system</div>",
                    unsafe_allow_html=True,
                )

    # ── Local image corpus ────────────────────────────────────────────────────
    st.write("")
    meta_path = Path("data/image_metadata.json")
    with st.container(border=True):
        _section_title("Training image corpus")
        if not meta_path.exists():
            st.caption(
                "Run `python scripts/download_images.py` to extract images from the OpenStax PDF."
            )
        else:
            metadata = json.loads(meta_path.read_text())
            images_dir = Path("data/images")
            # An image counts as available if its file is actually on disk.
            available = [m for m in metadata if (images_dir / m["filename"]).exists()]
            total_kb = sum(m.get("size_kb", 0) for m in available)

            c1, c2, c3 = st.columns(3)
            c1.metric("Images", len(available))
            c2.metric("Total size", f"{total_kb / 1024:.0f} MB")
            c3.metric("Source", "OpenStax A&P 2e")

            # Show grid of image info
            with st.expander(f"View all {len(available)} images"):
                for m in available[:200]:  # cap the list for performance
                    icon = "✅"
                    detail = (
                        m.get("region", "").replace("_", " ").title()
                        or f"page {m.get('page', '?')}"
                    )
                    st.caption(f"{icon} **{m['filename']}** · {detail} · {m.get('size_kb', 0)} KB")
