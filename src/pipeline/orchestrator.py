"""
orchestrator.py

Ties every agent together into one usable interview session:

    Resume/JD Matcher  ->  skill_gaps
            |
            v
    Next Question Agent  -->  question
            |
            v
    (candidate answers by voice)
            |
            v
    Whisper (Speech-to-Text)  -->  transcript
            |
            v
    Evaluation Agent (RAG + LLM)  -->  content score/feedback
    Confidence Agent  -->  delivery/confidence metrics
            |
            v
    Feedback Agent  -->  combined per-question feedback
            |
            v
    (repeat for N questions)
            |
            v
    Feedback Agent's build_summary_report()  -->  final interview report

Place this file at: src/pipeline/orchestrator.py

Usage (see the __main__ block at the bottom for a full runnable demo):

    from src.pipeline.orchestrator import InterviewSession

    session = InterviewSession(resume_index=0, jd_index=0)   # optional resume/JD targeting
    q = session.next_question()
    result = session.submit_answer(audio_path="path/to/answer.wav")
    ...
    summary = session.get_summary()
"""

import os
import pandas as pd
import whisper
from dotenv import load_dotenv

from src.agents.evaluation_agent import evaluate_answer
from src.agents.confidence_agent import analyze_delivery
from src.agents.feedback_agent import generate_feedback, build_summary_report
from src.agents.next_question_agent import pick_next_question
from src.agents.resume_jd_matcher import match_resume_to_jd, match_texts

load_dotenv()

QUESTIONS_PATH = "data/processed/text/unified_questions.csv"


# ---------------------------------------------------------------------------
# Whisper — lazy-loaded singleton, same pattern as retrieval.py's
# get_retriever() and resume_jd_matcher.py's _get_model(), so importing this
# module doesn't immediately load a ~500MB model into memory.
# ---------------------------------------------------------------------------
_whisper_model = None


def _get_whisper_model(size: str = "small"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(size)
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """Transcribes a .wav file to text using Whisper (small model, ~95.5%
    content accuracy on real spontaneous speech — see
    docs/Speech-to-Text Evaluation Report.md)."""
    model = _get_whisper_model()
    result = model.transcribe(audio_path)
    return result["text"].strip()


class InterviewSession:
    """
    Manages one full mock-interview session: question selection, answer
    processing (STT + evaluation + confidence + feedback), and the final
    summary report.
    """

    def __init__(self, resume_index: int = None, jd_index: int = None,
                 resume_text: str = None, jd_text: str = None,
                 job_title: str = "Uploaded JD", domain_mix: dict = None):
        """
        Args:
            resume_index / jd_index: row indices into unified_resumes.csv /
                unified_job_descriptions.csv — use for testing against the
                existing dataset.
            resume_text / jd_text: raw text of a REAL user-uploaded resume
                and job description — use this for the actual frontend, so
                users aren't limited to resumes/JDs already in the dataset.
                Takes priority over resume_index/jd_index if both are given.
            job_title: optional display label when using resume_text/jd_text.
            domain_mix: optional override for the HR/Technical question mix
                (see next_question_agent.DEFAULT_DOMAIN_MIX).
        """
        self.questions_df = pd.read_csv(QUESTIONS_PATH)
        if "question_id" not in self.questions_df.columns:
            # unified_questions.csv has no id column by default — see
            # next_question_agent.py's docstring for why this is needed.
            self.questions_df["question_id"] = self.questions_df.index.astype(str)

        self.history = []          # feeds pick_next_question()
        self.session_records = []  # feeds build_summary_report()
        self.domain_mix = domain_mix

        self.match_info = None
        self.skill_gaps = []
        if resume_text is not None and jd_text is not None:
            self.match_info = match_texts(resume_text, jd_text, job_title=job_title)
            self.skill_gaps = self.match_info.get("missing_skills", [])
            print(f"[orchestrator] Resume/JD match score: {self.match_info['match_score']}%")
            print(f"[orchestrator] Targeting skill gaps: {self.skill_gaps}")
        elif resume_index is not None and jd_index is not None:
            self.match_info = match_resume_to_jd(resume_index, jd_index)
            self.skill_gaps = self.match_info.get("missing_skills", [])
            print(f"[orchestrator] Resume/JD match score: {self.match_info['match_score']}%")
            print(f"[orchestrator] Targeting skill gaps: {self.skill_gaps}")

        self.current_question = None

    def next_question(self) -> dict:
        """Selects and returns the next question to ask. Raises ValueError
        (from pick_next_question) once the question bank is exhausted."""
        row = pick_next_question(
            history=self.history,
            resume={}, jd={},
            questions_df=self.questions_df,
            skill_gaps=self.skill_gaps,
            domain_mix=self.domain_mix,
        )
        self.current_question = row
        return row

    def submit_answer(self, audio_path: str = None, transcript: str = None) -> dict:
        """
        Processes the candidate's answer to the current question.

        Provide EITHER:
          - audio_path only: transcribed via Whisper, and also used for
            confidence/delivery analysis.
          - transcript only: skips Whisper (useful for text-only testing),
            no confidence/delivery metrics will be available.
          - both: transcript is used as-is (skips Whisper), audio_path is
            still used for confidence/delivery analysis.

        Returns:
        {
            "transcript": str,
            "evaluation_result": {...},   # from evaluate_answer()
            "confidence_result": {...} | None,  # from analyze_delivery()
            "feedback_result": {...}      # from generate_feedback()
        }
        """
        if self.current_question is None:
            raise RuntimeError("Call next_question() before submit_answer().")
        if audio_path is None and transcript is None:
            raise ValueError("Provide at least one of audio_path or transcript.")

        if transcript is None:
            transcript = transcribe_audio(audio_path)

        question_text = self.current_question.get("question")
        domain = self.current_question.get("domain", "HR")

        evaluation_result = evaluate_answer(question_text, transcript, domain=domain)

        confidence_result = None
        if audio_path is not None:
            confidence_result = analyze_delivery(audio_path, transcript)

        feedback_result = generate_feedback(evaluation_result, confidence_result)

        self.history.append({
            "question_id": self.current_question["question_id"],
            "domain": domain,
            "content_score": evaluation_result.get("content_score"),
        })
        self.session_records.append({
            "domain": domain,
            "evaluation_result": evaluation_result,
            "feedback_result": feedback_result,
        })

        self.current_question = None

        return {
            "transcript": transcript,
            "evaluation_result": evaluation_result,
            "confidence_result": confidence_result,
            "feedback_result": feedback_result,
        }

    def get_summary(self) -> dict:
        """Returns the final interview report (see feedback_agent.build_summary_report)."""
        return build_summary_report(self.session_records)


# ---------------------------------------------------------------------------
# Demo / smoke test — runs a 2-question mock session end-to-end.
# Uses a real RAVDESS audio file for confidence/delivery metrics, paired
# with a realistic hand-written transcript (skipping Whisper transcription
# for question 1, since RAVDESS clips just say scripted sentences that
# aren't real interview answers — but still exercising the full
# evaluation + confidence + feedback pipeline on real audio features).
# Question 2 demonstrates Whisper actually transcribing the same clip, to
# confirm that code path works too, even though the "answer" content won't
# make sense for the question (expected — it's a fixed demo sentence).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob

    session = InterviewSession()  # no resume/JD targeting for this basic demo

    sample_audio_candidates = glob.glob("data/raw/audio_samples/**/*.wav", recursive=True)
    sample_audio = sample_audio_candidates[0] if sample_audio_candidates else None

    print("=" * 70)
    print("DEMO — Question 1 (transcript provided directly, real audio for confidence)")
    q1 = session.next_question()
    print(f"Q: {q1['question']}  (domain={q1['domain']})")

    result1 = session.submit_answer(
        audio_path=sample_audio,
        transcript=(
            "In my last group project, we were behind schedule with two weeks "
            "left, so I proposed splitting into smaller sub-teams and ran daily "
            "check-ins to track blockers. We ended up finishing on time."
        ),
    )
    print(f"Transcript used: {result1['transcript']}")
    print(f"Content score: {result1['evaluation_result'].get('content_score')}")
    print(f"Confidence: {result1['confidence_result']}")
    print(f"Feedback: {result1['feedback_result']['feedback_text']}")

    print("\n" + "=" * 70)
    print("DEMO — Question 2 (Whisper actually transcribes the audio)")
    q2 = session.next_question()
    print(f"Q: {q2['question']}  (domain={q2['domain']})")

    if sample_audio:
        result2 = session.submit_answer(audio_path=sample_audio)  # transcript=None -> Whisper runs
        print(f"Whisper transcript: {result2['transcript']}")
        print(f"Content score: {result2['evaluation_result'].get('content_score')}")
        print(f"Feedback: {result2['feedback_result']['feedback_text']}")
    else:
        print("No sample audio found under data/raw/audio_samples — skipping Whisper demo.")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print(session.get_summary())