# Evaluation Agent — Interface for Person 2 (Feedback Agent)

This document describes exactly what `evaluate_answer()` expects as input and
what it returns, so the Feedback Agent can be built against a stable contract
right now — no need to wait.

---

## How to import it

```python
from src.agents.evaluation_agent import evaluate_answer
```

Make sure `GEMINI_API_KEY` is set in your own `.env` file (see main README /
setup guide) — you'll need your own free Gemini API key from
https://aistudio.google.com to actually call this function.

---

## Function signature

```python
evaluate_answer(question: str, user_answer: str, domain: str = "HR", top_k: int = 2) -> dict
```

### Parameters
| Name | Type | Required | Notes |
|---|---|---|---|
| `question` | str | Yes | The interview question that was asked |
| `user_answer` | str | Yes | The candidate's transcribed answer (from Speech-to-Text) |
| `domain` | str | No (default `"HR"`) | Either `"HR"` or `"Technical"` — must match the type of question asked |
| `top_k` | int | No (default `2`) | How many reference Q&A pairs to retrieve for grounding — usually no need to change this |

### Return value — always a `dict` with these keys:

```python
{
  "content_score": 7,                     # int, 1-10, or None if parsing failed
  "strengths": "...",                     # str, 1-2 sentences
  "missing_points": "...",                # str, 1-2 sentences
  "structure_feedback": "..."             # str, 1 sentence
}
```

### Error case
If the LLM response couldn't be parsed as valid JSON (rare, but possible), the function returns this shape instead — **always check for the `"error"` key before assuming the normal fields are populated**:

```python
{
  "content_score": None,
  "strengths": None,
  "missing_points": None,
  "structure_feedback": None,
  "error": "Could not parse LLM response as JSON",
  "raw_response": "<the raw text that failed to parse>"
}
```

---

## Example usage (what your Feedback Agent code will look like)

```python
from src.agents.evaluation_agent import evaluate_answer

result = evaluate_answer(
    question="Tell me about a time you showed leadership",
    user_answer="During my final year project, our team was behind schedule...",
    domain="HR"
)

if "error" in result:
    # handle gracefully — e.g. ask user to retry, or fall back to a generic message
    print("Evaluation failed, raw response was:", result["raw_response"])
else:
    print(f"Score: {result['content_score']}/10")
    print(f"Strengths: {result['strengths']}")
    print(f"Missing: {result['missing_points']}")
    print(f"Structure: {result['structure_feedback']}")
```

---

## Stub version (use this if you want to build/test before setting up your own Gemini key)

Drop this at the top of your own working file if you want to develop against
fake data first, then delete it and use the real import once you're ready:

```python
def evaluate_answer(question, user_answer, domain="HR", top_k=2):
    """TEMPORARY STUB — replace with:
    from src.agents.evaluation_agent import evaluate_answer
    once you're ready to test against the real thing."""
    return {
        "content_score": 7,
        "strengths": "Clear structure and relevant example provided.",
        "missing_points": "Could include more specific, measurable outcomes.",
        "structure_feedback": "Follows a logical flow, slightly rushed at the end."
    }
```

---

## Notes / known behavior
- Technical question coverage in the underlying dataset is currently thin (~140 questions) — retrieval still works correctly via domain filtering, but expect less variety on `domain="Technical"` calls for now. This is a known, tracked limitation (see `docs/data_preprocessing_report.md`), not a bug in this function.
- Each call makes a live API request to Gemini — expect ~1-3 seconds of latency per call. If you're testing in a loop, consider caching results or using the stub above during early development.