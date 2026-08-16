"""
evaluation_agent.py

The Evaluation Agent: given a question and a candidate's answer, retrieves
grounding context (RAG) from the question bank and asks an LLM (Groq —
Llama 3.3 70B) to score the answer's content quality, structure, and
completeness.

NOTE: Originally built on Google Gemini's free tier, but migrated to Groq
after repeatedly hitting Gemini's 20 requests/day free-tier quota (429
errors) and occasional server overload (503 errors) during development.
Groq's free tier (30 requests/min, ~1,000/day as of mid-2026) is far more
generous and better suited to iterative dev/demo use.

Place this file at: src/agents/evaluation_agent.py
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

from src.data_processing.retrieval import get_retriever

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Llama 3.3 70B — strong general reasoning, generous free tier on Groq.
# Swap here if you want to try a different Groq-hosted model.
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"


def _build_prompt(question: str, user_answer: str, context: str) -> str:
    return f"""You are an expert interview coach evaluating a candidate's answer.

QUESTION ASKED: {question}

CANDIDATE'S ANSWER: {user_answer}

REFERENCE MATERIAL (similar questions with ideal answers, for grounding your judgment):
{context}

Evaluate the candidate's answer and respond ONLY in this exact JSON format, no other text, no markdown code blocks:
{{
  "content_score": <integer 1-10>,
  "strengths": "<1-2 sentences on what was good>",
  "missing_points": "<1-2 sentences on what was missing or could improve>",
  "structure_feedback": "<1 sentence on clarity/structure of the answer>"
}}"""


def _parse_llm_json(raw_text: str) -> dict:
    """Strips markdown code fences (some models add them) and parses JSON,
    with a safe fallback if parsing fails so the app doesn't crash on a bad response."""
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "content_score": None,
            "strengths": None,
            "missing_points": None,
            "structure_feedback": None,
            "error": "Could not parse LLM response as JSON",
            "raw_response": raw_text,
        }


def evaluate_answer(question: str, user_answer: str, domain: str = "HR", top_k: int = 2) -> dict:
    """
    Evaluates a candidate's spoken/written answer to an interview question.

    Args:
        question: the interview question that was asked
        user_answer: the candidate's answer (e.g. from Speech-to-Text transcript)
        domain: "HR" or "Technical" — determines which part of the question bank
                is used for retrieval grounding
        top_k: number of reference Q&A pairs to retrieve for grounding

    Returns:
        dict with keys: content_score, strengths, missing_points, structure_feedback
        (or an "error" key + raw_response if the LLM output couldn't be parsed)
    """
    retriever = get_retriever()
    retrieved = retriever.retrieve(question, top_k=top_k, domain=domain)

    context = "\n\n".join(
        f"Reference Question: {row['question']}\nIdeal Answer: {row['answer']}"
        for _, row in retrieved.iterrows()
    )

    prompt = _build_prompt(question, user_answer, context)
    response = _client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    return _parse_llm_json(response.choices[0].message.content.strip())


if __name__ == "__main__":
    # Quick manual test when running this file directly
    result = evaluate_answer(
        "Tell me about a time you had to work under pressure",
        "Once I had a project deadline that got moved up. I stayed organized by "
        "making a task list and communicating with my team about priorities. "
        "We finished on time.",
        domain="HR",
    )
    print(json.dumps(result, indent=2))