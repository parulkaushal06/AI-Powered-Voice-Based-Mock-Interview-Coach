"""
Feedback Agent
--------------
Owner: Person 2

Takes Evaluation Agent's output (Person 1, evaluate_answer()) + optional
Confidence/Communication output (Person 3, analyze_delivery()) and turns
them into a natural, encouraging feedback paragraph.

Matches evaluation_agent_interface.md exactly:

evaluate_answer() returns EITHER:
    {
        "content_score": 7,               # int 1-10, or None
        "strengths": "...",                # str
        "missing_points": "...",           # str
        "structure_feedback": "..."        # str
    }
OR, on parse failure:
    {
        "content_score": None, "strengths": None,
        "missing_points": None, "structure_feedback": None,
        "error": "Could not parse LLM response as JSON",
        "raw_response": "<raw text>"
    }

confidence_result — see CONFIDENCE_AGENT_INTERFACE.md for the real shape:
    {
        "confidence_label": "confident",        # "confident" / "nervous"
        "confidence_scores": {"confident": 0.92, "nervous": 0.08},
        "pace_pause_metrics": {"wpm": 142, "pause_ratio": 0.18, ...}
    }
_adapt_confidence_result() below converts this into the flatter shape
_summarize_delivery() expects.

NOTE: migrated from Gemini to Groq (see evaluation_agent.py for why —
Gemini's free tier hit both 429 quota-exceeded and 503 server-overload
errors repeatedly during dev). Uses the same GROQ_API_KEY as
evaluation_agent.py.
"""

from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# LLM call — uses Groq (Llama 3.3 70B), same provider as Person 1's
# evaluation_agent.py, for consistency and to share one free-tier quota
# budget. Falls back to a plain template if no key is set / call fails, so
# the pipeline never crashes during dev/demo.
# ---------------------------------------------------------------------------
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"  # same model Person 1 uses


def call_llm(prompt: str, system: Optional[str] = None) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _fallback_template()

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[feedback_agent] LLM call failed, using fallback: {e}")
        return _fallback_template()


def _fallback_template() -> str:
    return ("Feedback generation is running in offline/template mode "
            "(GROQ_API_KEY not set or call failed). Set your key in .env "
            "to get real, tailored feedback text.")


# ---------------------------------------------------------------------------
# Adapter: converts analyze_delivery()'s real output shape (see
# CONFIDENCE_AGENT_INTERFACE.md) into the flatter shape _summarize_delivery()
# expects. Keeps the summarization logic simple without needing to change it
# every time Person 3's output format changes.
# ---------------------------------------------------------------------------
def _adapt_confidence_result(raw: Optional[dict]) -> Optional[dict]:
    if not raw:
        return None
    return {
        "confidence_score": raw.get("confidence_scores", {}).get("confident"),
        "pace_wpm": raw.get("pace_pause_metrics", {}).get("wpm"),
        "pause_ratio": raw.get("pace_pause_metrics", {}).get("pause_ratio"),
    }


# ---------------------------------------------------------------------------
# Helpers that translate raw evaluation fields into short natural-language
# fragments fed to the LLM prompt
# ---------------------------------------------------------------------------
def _summarize_content(evaluation_result: dict) -> str:
    if evaluation_result.get("error"):
        return ("Evaluation could not be scored automatically this time "
                "(parsing error on our end, not the candidate's fault).")

    score = evaluation_result.get("content_score")
    strengths = evaluation_result.get("strengths") or "none noted"
    missing = evaluation_result.get("missing_points") or "none noted"
    structure = evaluation_result.get("structure_feedback") or "not noted"

    score_str = f"{score}/10" if score is not None else "not scored"
    return (f"Content score: {score_str}. "
            f"Strengths: {strengths} "
            f"Missing points: {missing} "
            f"Structure: {structure}")


def _summarize_delivery(confidence_result: Optional[dict]) -> str:
    if not confidence_result:
        return "No delivery/audio metrics available for this answer."

    conf = confidence_result.get("confidence_score")
    wpm = confidence_result.get("pace_wpm")
    pause_ratio = confidence_result.get("pause_ratio")

    parts = []
    if conf is not None:
        level = "confident" if conf >= 0.65 else ("neutral" if conf >= 0.4 else "hesitant")
        parts.append(f"vocal confidence came across as {level} ({conf:.2f})")
    if wpm is not None:
        pace = "fast" if wpm > 160 else ("slow" if wpm < 110 else "well-paced")
        parts.append(f"speaking pace was {pace} ({wpm} wpm)")
    if pause_ratio is not None:
        pauses = "frequent long pauses" if pause_ratio > 0.25 else "normal pausing"
        parts.append(pauses)

    return "Delivery: " + ", ".join(parts) + "." if parts else "Delivery metrics incomplete."


# ---------------------------------------------------------------------------
# Main entry point Person 4's orchestrator calls
# ---------------------------------------------------------------------------
def generate_feedback(evaluation_result: dict, confidence_result: Optional[dict] = None) -> dict:
    """
    Returns:
    {
        "feedback_text": str,
        "content_score": int | None,
        "confidence_score": float | None,
        "domain_ok": bool     # False if evaluation_result had an "error"
    }
    """
    confidence_result = _adapt_confidence_result(confidence_result)

    # Graceful handling of the evaluation_agent error case (per interface doc)
    if evaluation_result.get("error"):
        return {
            "feedback_text": (
                "We couldn't automatically score that answer this time — "
                "this is a technical hiccup on our end, not a reflection of "
                "your answer. Feel free to move on to the next question."
            ),
            "content_score": None,
            "confidence_score": (confidence_result or {}).get("confidence_score"),
            "domain_ok": False,
        }

    content_summary = _summarize_content(evaluation_result)
    delivery_summary = _summarize_delivery(confidence_result)

    system_prompt = (
        "You are a supportive but honest interview coach. Given structured "
        "notes about a candidate's answer (content quality + delivery), write "
        "ONE short encouraging feedback paragraph (4-6 sentences). Be "
        "specific about what to fix, never generic — use the strengths, "
        "missing points, and structure notes given to you rather than "
        "just repeating the score. End with one concrete tip for the next "
        "answer."
    )
    user_prompt = f"{content_summary}\n{delivery_summary}\n\nWrite the feedback paragraph now."

    feedback_text = call_llm(user_prompt, system=system_prompt)

    return {
        "feedback_text": feedback_text,
        "content_score": evaluation_result.get("content_score"),
        "confidence_score": (confidence_result or {}).get("confidence_score"),
        "domain_ok": True,
    }


# ---------------------------------------------------------------------------
# Final interview summary report — built once at session end from the list
# of (evaluation_result, feedback_result, domain) collected per question
# ---------------------------------------------------------------------------
def build_summary_report(session_records: list) -> dict:
    """
    session_records: list of dicts, one per question answered, each shaped:
        {
            "domain": "HR" | "Technical",
            "evaluation_result": {...},   # raw output of evaluate_answer()
            "feedback_result": {...}      # output of generate_feedback()
        }

    Returns:
    {
        "avg_content_score": float,
        "avg_confidence_score": float | None,
        "strengths": [...],            # domains where avg score >= 7
        "weaknesses": [...],           # domains where avg score < 5
        "confidence_trend": [...],     # scores in the order answered
        "domain_breakdown": {"HR": avg, "Technical": avg},
        "role_fit_note": str,
        "unscored_answers": int        # count of evaluation errors, for transparency
    }
    """
    if not session_records:
        return {"error": "No answers recorded for this session."}

    scored = [r for r in session_records
              if r["evaluation_result"].get("content_score") is not None]
    unscored = len(session_records) - len(scored)

    content_scores = [r["evaluation_result"]["content_score"] for r in scored]
    conf_scores = [r["feedback_result"].get("confidence_score") for r in session_records
                   if r["feedback_result"].get("confidence_score") is not None]

    domain_scores = {}
    for r in scored:
        domain = r.get("domain", "HR")
        domain_scores.setdefault(domain, []).append(r["evaluation_result"]["content_score"])
    domain_breakdown = {d: round(sum(v) / len(v), 2) for d, v in domain_scores.items()}

    avg_content = round(sum(content_scores) / len(content_scores), 2) if content_scores else 0
    avg_conf = round(sum(conf_scores) / len(conf_scores), 2) if conf_scores else None

    strengths = [d for d, avg in domain_breakdown.items() if avg >= 7]
    weaknesses = [d for d, avg in domain_breakdown.items() if avg < 5]

    if avg_content >= 7.5:
        role_fit_note = "Strong overall performance — close to role-ready."
    elif avg_content >= 5:
        role_fit_note = "Moderate fit — a few gaps to close before real interviews."
    else:
        role_fit_note = "Significant preparation still needed on core content."

    return {
        "avg_content_score": avg_content,
        "avg_confidence_score": avg_conf,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "confidence_trend": conf_scores,
        "domain_breakdown": domain_breakdown,
        "role_fit_note": role_fit_note,
        "unscored_answers": unscored,
    }


if __name__ == "__main__":
    # Smoke test using the stub shape from evaluation_agent_interface.md,
    # and the REAL analyze_delivery() shape for confidence (see
    # CONFIDENCE_AGENT_INTERFACE.md) to exercise _adapt_confidence_result().
    demo_eval = {
        "content_score": 7,
        "strengths": "Clear structure and relevant example provided.",
        "missing_points": "Could include more specific, measurable outcomes.",
        "structure_feedback": "Follows a logical flow, slightly rushed at the end.",
    }
    demo_conf = {
        "confidence_label": "nervous",
        "confidence_scores": {"confident": 0.45, "nervous": 0.55},
        "pace_pause_metrics": {"wpm": 175, "pause_ratio": 0.3},
    }
    result = generate_feedback(demo_eval, demo_conf)
    print(result)

    demo_error_eval = {
        "content_score": None, "strengths": None, "missing_points": None,
        "structure_feedback": None, "error": "Could not parse LLM response as JSON",
        "raw_response": "garbage text",
    }
    print(generate_feedback(demo_error_eval))