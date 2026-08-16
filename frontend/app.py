"""
AI Interview Coach — Streamlit Frontend

Run with:
    streamlit run frontend/app.py

Flow:
    1. User optionally pastes their resume + target job description
       (used for skill-gap-targeted questioning via resume_jd_matcher.py)
    2. User starts the interview
    3. For each question: record/upload an answer -> see transcript,
       content score, confidence, and combined feedback
    4. User ends the session -> sees the final summary report
"""

import os
import sys
import tempfile

import streamlit as st

# Make sure the project root is on sys.path so `from src...` imports work
# regardless of the directory Streamlit is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.orchestrator import InterviewSession

st.set_page_config(page_title="AI Interview Coach", page_icon="🎤", layout="centered")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------
if "session" not in st.session_state:
    st.session_state.session = None
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "finished" not in st.session_state:
    st.session_state.finished = False
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

st.title("🎤 AI Interview Coach")
st.caption("Practice mock interviews with AI-driven questions, evaluation, and feedback.")


# ---------------------------------------------------------------------------
# SCREEN 1 — Setup (before session starts)
# ---------------------------------------------------------------------------
if st.session_state.session is None and not st.session_state.finished:
    st.subheader("1. (Optional) Personalize your interview")
    st.write(
        "Paste your resume and a target job description to have questions "
        "targeted at your actual skill gaps. Leave blank for a general mix "
        "of HR and technical questions."
    )

    resume_text = st.text_area("Your resume (plain text)", height=150,
                                placeholder="Paste your resume text here...")
    jd_text = st.text_area("Target job description (plain text)", height=150,
                            placeholder="Paste the job description here...")
    job_title = st.text_input("Job title (optional label)", value="")

    st.subheader("2. Start")
    num_questions = st.slider("Number of questions", min_value=1, max_value=10, value=3)

    if st.button("Start Interview", type="primary"):
        with st.spinner("Setting up your session..."):
            if resume_text.strip() and jd_text.strip():
                session = InterviewSession(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    job_title=job_title or "Uploaded JD",
                )
                if session.match_info:
                    st.success(
                        f"Resume/JD match score: {session.match_info['match_score']}% — "
                        f"targeting: {', '.join(session.skill_gaps) or 'no specific gaps found'}"
                    )
            else:
                session = InterviewSession()

        st.session_state.session = session
        st.session_state.target_questions = num_questions
        st.session_state.question_count = 0
        st.session_state.current_question = session.next_question()
        st.rerun()


# ---------------------------------------------------------------------------
# SCREEN 2 — Active interview (question -> answer -> feedback loop)
# ---------------------------------------------------------------------------
elif st.session_state.session is not None and not st.session_state.finished:
    session = st.session_state.session
    q = st.session_state.current_question

    st.subheader(f"Question {st.session_state.question_count + 1} of {st.session_state.target_questions}")
    st.info(f"**[{q['domain']}]** {q['question']}")

    if st.session_state.last_result is None:
        # --- Waiting for an answer ---
        st.write("Record your answer:")
        audio_value = st.audio_input("Record your answer")

        st.write("— or upload a .wav file instead:")
        uploaded_file = st.file_uploader("Upload answer audio", type=["wav"])

        audio_bytes = None
        if audio_value is not None:
            audio_bytes = audio_value.read()
        elif uploaded_file is not None:
            audio_bytes = uploaded_file.read()

        if audio_bytes and st.button("Submit Answer", type="primary"):
            with st.spinner("Transcribing and evaluating your answer..."):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                result = session.submit_answer(audio_path=tmp_path)
                os.unlink(tmp_path)

            st.session_state.last_result = result
            st.rerun()

    else:
        # --- Showing feedback for the answer just submitted ---
        result = st.session_state.last_result

        st.write("**Your answer (transcribed):**")
        st.write(result["transcript"])

        col1, col2 = st.columns(2)
        with col1:
            score = result["evaluation_result"].get("content_score")
            st.metric("Content Score", f"{score}/10" if score is not None else "N/A")
        with col2:
            if result["confidence_result"]:
                label = result["confidence_result"]["confidence_label"]
                conf_pct = result["confidence_result"]["confidence_scores"].get(label, 0) * 100
                st.metric("Delivery", label.capitalize(), f"{conf_pct:.0f}% confidence")
            else:
                st.metric("Delivery", "N/A")

        st.write("**Feedback:**")
        st.write(result["feedback_result"]["feedback_text"])

        with st.expander("See detailed breakdown"):
            st.json(result["evaluation_result"])
            if result["confidence_result"]:
                st.json(result["confidence_result"])

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.session_state.question_count + 1 < st.session_state.target_questions:
                if st.button("Next Question →", type="primary"):
                    st.session_state.question_count += 1
                    st.session_state.current_question = session.next_question()
                    st.session_state.last_result = None
                    st.rerun()
        with col_b:
            if st.button("End Interview"):
                st.session_state.finished = True
                st.rerun()

        if st.session_state.question_count + 1 >= st.session_state.target_questions:
            st.info("That was your last question. Click **End Interview** to see your summary.")


# ---------------------------------------------------------------------------
# SCREEN 3 — Final summary report
# ---------------------------------------------------------------------------
elif st.session_state.finished:
    st.subheader("📊 Your Interview Summary")

    summary = st.session_state.session.get_summary()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Content Score", f"{summary.get('avg_content_score', 'N/A')}/10")
    with col2:
        conf = summary.get("avg_confidence_score")
        st.metric("Average Confidence", f"{conf*100:.0f}%" if conf is not None else "N/A")

    st.write(f"**Overall assessment:** {summary.get('role_fit_note', '')}")

    if summary.get("strengths"):
        st.success(f"**Strengths:** {', '.join(summary['strengths'])}")
    if summary.get("weaknesses"):
        st.warning(f"**Areas to improve:** {', '.join(summary['weaknesses'])}")

    st.write("**Score by domain:**")
    st.bar_chart(summary.get("domain_breakdown", {}))

    if st.session_state.session.match_info:
        st.write("**Resume/JD match info:**")
        st.json(st.session_state.session.match_info)

    st.divider()
    if st.button("Start New Interview"):
        st.session_state.session = None
        st.session_state.current_question = None
        st.session_state.last_result = None
        st.session_state.finished = False
        st.session_state.question_count = 0
        st.rerun()