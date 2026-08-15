"""
Next Question Agent
--------------------
Owner: Person 2

Given the resume/JD, question history, and how previous answers scored,
pick the next question from unified_questions.csv — mixing HR/Technical
based on role and adapting difficulty to performance.

IMPORTANT: evaluate_answer() takes domain: "HR" | "Technical" (see
evaluation_agent_interface.md). This picker outputs a question whose
domain you should pass straight into evaluate_answer(domain=...) so the
two stay in sync.

Assumed unified_questions.csv columns — CONFIRM these against the real
file and adjust DOMAIN_COL / DIFFICULTY_COL / SKILL_COL below if the
actual names differ (this is the only part of the file you'd need to touch):
    question_id, question_text, domain ("HR"/"Technical"),
    difficulty ("easy"/"medium"/"hard"), skill_tag, ideal_answer

NOTE: unified_questions.csv does NOT have a question_id column by default —
add one when loading it, e.g.:
    questions_df = pd.read_csv("data/processed/text/unified_questions.csv")
    questions_df["question_id"] = questions_df.index.astype(str)

Assumed history item shape (one per question already asked), built from
evaluate_answer()'s own return value:
    {
        "question_id": "...",
        "domain": "HR",              # what you asked evaluate_answer() with
        "content_score": 6           # evaluation_result["content_score"], may be None on error
    }
"""

import pandas as pd
import random

DOMAIN_COL = "domain"
DIFFICULTY_COL = "difficulty"
SKILL_COL = "category"

# HR/Technical only, per evaluate_answer()'s domain param. Skews technical
# since that's the stated core value-add — tune with your team.
DEFAULT_DOMAIN_MIX = {"Technical": 0.6, "HR": 0.4}


def _asked_ids(history: list) -> set:
    return {h["question_id"] for h in history}


def _recent_avg_score(history: list, n: int = 2) -> float:
    scored = [h.get("content_score") for h in history if h.get("content_score") is not None]
    if not scored:
        return 5.0  # neutral starting point, or all recent answers had eval errors
    recent = scored[-n:]
    return sum(recent) / len(recent)


def _next_domain(history: list, domain_mix: dict = DEFAULT_DOMAIN_MIX) -> str:
    """Pick whichever domain is currently under-represented vs. the target mix."""
    if not history:
        return max(domain_mix, key=domain_mix.get)

    counts = {d: 0 for d in domain_mix}
    for h in history:
        d = h.get("domain", "HR")
        counts[d] = counts.get(d, 0) + 1

    total = len(history)
    deficit = {d: target - (counts.get(d, 0) / total) for d, target in domain_mix.items()}
    return max(deficit, key=deficit.get)


def _target_difficulty(history: list) -> str:
    """Adaptive difficulty: do well -> go harder, struggle -> ease off."""
    avg = _recent_avg_score(history)
    if avg >= 7.5:
        return "hard"
    elif avg >= 4.5:
        return "medium"
    else:
        return "easy"


def pick_next_question(history: list, resume: dict, jd: dict,
                        questions_df: pd.DataFrame,
                        skill_gaps: list = None,
                        domain_mix: dict = None) -> dict:
    """
    Returns a single row (as dict) from questions_df — the next question to
    ask. Also returns "domain" so the caller can pass it straight into
    evaluate_answer(domain=...).

    Priority order:
      1. Not already asked
      2. Matches the currently under-represented domain (HR/Technical)
      3. Matches the adaptive difficulty target
      4. If skill_gaps provided (from Person 4's resume/JD matcher), prefer
         questions tagged with a gap skill
      5. Random pick among whatever's left, to avoid repetitive ordering
    """
    domain_mix = domain_mix or DEFAULT_DOMAIN_MIX
    asked = _asked_ids(history)
    pool = questions_df[~questions_df["question_id"].isin(asked)].copy()

    if pool.empty:
        raise ValueError("No unasked questions remain in unified_questions.csv")

    target_domain = _next_domain(history, domain_mix)
    target_difficulty = _target_difficulty(history)

    def _filter(df, **kwargs):
        f = df
        for col, val in kwargs.items():
            if val is not None and col in f.columns:
                narrowed = f[f[col] == val]
                if not narrowed.empty:
                    f = narrowed
        return f

    candidates = _filter(pool, **{DOMAIN_COL: target_domain, DIFFICULTY_COL: target_difficulty})

    if skill_gaps and SKILL_COL in candidates.columns:
        # Substring match (not exact equality) since skill_gaps comes from
        # resume_jd_matcher.py using tech skill names (e.g. "SQL", "AWS")
        # while unified_questions.csv's category column uses broader labels
        # (e.g. "Database and SQL"). Case-insensitive substring matching
        # bridges the two vocabularies without needing a hand-maintained
        # mapping table. Not a guaranteed 1:1 match for every skill —
        # documented as a known limitation.
        gap_lower = [str(g).lower() for g in skill_gaps]
        cat_lower = candidates[SKILL_COL].astype(str).str.lower()
        mask = cat_lower.apply(lambda cat: any(g in cat or cat in g for g in gap_lower))
        gap_matches = candidates[mask]
        if not gap_matches.empty:
            candidates = gap_matches

    if candidates.empty:
        candidates = pool  # last-resort fallback

    chosen = candidates.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0].to_dict()
    chosen.setdefault("domain", chosen.get(DOMAIN_COL, target_domain))
    return chosen


if __name__ == "__main__":
    demo_df = pd.DataFrame([
        {"question_id": "q1", "question_text": "Explain REST vs GraphQL", "domain": "Technical", "difficulty": "medium", "category": "api", "ideal_answer": "..."},
        {"question_id": "q2", "question_text": "Tell me about a time you showed leadership", "domain": "HR", "difficulty": "easy", "category": "communication", "ideal_answer": "..."},
        {"question_id": "q3", "question_text": "Why do you want this role?", "domain": "HR", "difficulty": "easy", "category": "motivation", "ideal_answer": "..."},
        {"question_id": "q4", "question_text": "Explain ACID properties in SQL", "domain": "Technical", "difficulty": "easy", "category": "Database and SQL", "ideal_answer": "..."},
    ])
    demo_history = [{"question_id": "q1", "domain": "Technical", "content_score": 8}]

    # Demo using a real resume_jd_matcher.py-style skill_gaps list
    nxt = pick_next_question(demo_history, resume={}, jd={}, questions_df=demo_df, skill_gaps=["SQL", "AWS"])
    print("Picked (with skill_gaps=['SQL', 'AWS']):")
    print(nxt)