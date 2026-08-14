import pandas as pd
import time
from src.agents.feedback_agent import generate_feedback, build_summary_report
from src.agents.next_question_agent import pick_next_question
from src.agents.evaluation_agent import evaluate_answer

questions_df = pd.read_csv("data/processed/text/unified_questions.csv")
questions_df = questions_df.reset_index().rename(columns={"index": "question_id"})
questions_df["question_id"] = questions_df["question_id"].astype(str)
questions_df = questions_df.rename(columns={"question": "question_text"})

history = []
session_records = []

for i in range(2):
    next_q = pick_next_question(history, resume={}, jd={}, questions_df=questions_df)
    print(f"\n--- Question {i+1}: {next_q['question_text']} (domain={next_q['domain']}) ---")

    fake_answer = "During my final year project, our team was behind schedule so I reorganized the sprint plan..."

    eval_result = None
    for attempt in range(3):
        try:
            eval_result = evaluate_answer(next_q["question_text"], fake_answer, domain=next_q["domain"])
            break
        except Exception as e:
            print(f"evaluate_answer failed (attempt {attempt+1}/3): {e}")
            time.sleep(15)

    if eval_result is None:
        eval_result = {"content_score": None, "strengths": [], "missing_points": [], "structure_feedback": ""}

    fake_confidence = {"confidence_score": 0.6, "pace_wpm": 150, "pause_ratio": 0.2}
    feedback_result = generate_feedback(eval_result, fake_confidence)

    print("Feedback:", feedback_result["feedback_text"])

    history.append({
        "question_id": next_q["question_id"],
        "domain": next_q["domain"],
        "content_score": eval_result.get("content_score"),
    })
    session_records.append({
        "domain": next_q["domain"],
        "evaluation_result": eval_result,
        "feedback_result": feedback_result,
    })

print("\n=== FINAL SUMMARY REPORT ===")
print(build_summary_report(session_records))