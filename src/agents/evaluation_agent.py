"""
evaluation_agent.py

The Evaluation Agent: given a question and a candidate's answer, retrieves
grounding context (RAG) from the question bank and asks an LLM (Gemini) to
score the answer's content quality, structure, and completeness.

Place this file at: src/agents/evaluation_agent.py
"""

import os
import json
from google import genai
from dotenv import load_dotenv

from src.data_processing.retrieval import get_retriever

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GEMINI_MODEL_NAME = "gemini-flash-latest"


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
    retriever = get_retriever()
    retrieved = retriever.retrieve(question, top_k=top_k, domain=domain)

    context = "\n\n".join(
        f"Reference Question: {row['question']}\nIdeal Answer: {row['answer']}"
        for _, row in retrieved.iterrows()
    )

    prompt = _build_prompt(question, user_answer, context)
    response = _client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )

    return _parse_llm_json(response.text.strip())


if __name__ == "__main__":
    result = evaluate_answer(
        "Tell me about a time you had to work under pressure",
        "Once I had a project deadline that got moved up. I stayed organized by "
        "making a task list and communicating with my team about priorities. "
        "We finished on time.",
        domain="HR",
    )
    print(json.dumps(result, indent=2))