import ast
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS_LIST = [
    "Python", "Java", "C++", "SQL", "Machine Learning",
    "Deep Learning", "TensorFlow", "PyTorch", "AWS",
    "Docker", "Kubernetes", "Linux", "Git", "Power BI",
    "Excel", "Hadoop", "Spark", "Hive", "Tableau",
    "NLP", "Flask", "Django", "MongoDB", "MySQL",
    "ETL", "R"
]

SKILL_ALIASES = {
    "python": ["python"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "sql": ["sql"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"],
    "aws": ["aws", "amazon web services"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "linux": ["linux"],
    "git": ["git", "github"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "ms excel", "microsoft excel"],
    "hadoop": ["hadoop"],
    "spark": ["spark", "apache spark"],
    "hive": ["hive", "apache hive"],
    "tableau": ["tableau"],
    "nlp": ["nlp", "natural language processing"],
    "flask": ["flask"],
    "django": ["django"],
    "mongodb": ["mongodb", "mongo db", "mongo"],
    "mysql": ["mysql"],
    "etl": ["etl", "extract transform load"],
    "r": ["r programming", "r language"]
}


def normalize_skill_text(text):
    """Normalize text for reliable skill matching."""
    text = str(text).lower()
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.replace("powerbi", "power bi")
    text = text.replace("microsoft excel", "excel")
    text = text.replace("ms excel", "excel")
    text = " ".join(text.split())
    return text


def skill_present(skill, text):
    """Check whether a skill or one of its aliases exists in text."""
    normalized_text = normalize_skill_text(text)

    aliases = SKILL_ALIASES.get(
        skill.lower(),
        [skill.lower()]
    )

    for alias in aliases:
        normalized_alias = normalize_skill_text(alias)

        if skill.lower() == "r":
            words = normalized_text.split()
            if "r" in words or normalized_alias in words:
                return True
        elif normalized_alias in normalized_text:
            return True

    return False


model = SentenceTransformer("all-MiniLM-L6-v2")


def load_data():
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    base_path = os.path.join(
        project_root,
        "data",
        "processed",
        "text"
    )

    resume_path = os.path.join(
        base_path,
        "unified_resumes.csv"
    )

    jd_path = os.path.join(
        base_path,
        "unified_job_descriptions.csv"
    )

    print("Loading resume data from:", resume_path)
    print("Loading JD data from:", jd_path)

    resume_df = pd.read_csv(resume_path)
    jd_df = pd.read_csv(jd_path)

    return resume_df, jd_df
resume_df, jd_df = load_data()

def match_resume_to_jd(resume_index, jd_index):
    resume_text = str(
        resume_df.iloc[resume_index]["raw_text"]
    )
    jd_text = str(
        jd_df.iloc[jd_index]["description"]
    )

    # Semantic similarity
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text)

    score = cosine_similarity(
        [resume_embedding],
        [jd_embedding]
    )[0][0] * 100

    # Resume skills
    raw_resume_skills = resume_df.iloc[
        resume_index
    ]["skills"]

    try:
        resume_skills = ast.literal_eval(
            str(raw_resume_skills)
        )
    except (ValueError, SyntaxError):
        resume_skills = []

    resume_skill_text = " ".join(
        str(skill) for skill in resume_skills
    )

    # Extract required skills from JD
    jd_skills = [
        skill for skill in SKILLS_LIST
        if skill_present(skill, jd_text)
    ]

    # Match resume skills with JD requirements
    matched = [
        skill for skill in jd_skills
        if skill_present(skill, resume_skill_text)
    ]

    missing = [
        skill for skill in jd_skills
        if not skill_present(skill, resume_skill_text)
    ]

    return {
        "resume_id": resume_df.iloc[resume_index]["resume_id"],
        "job_title": jd_df.iloc[jd_index]["job_title"],
        "match_score": round(float(score), 2),
        "matched_skills": matched,
        "missing_skills": missing
    }


if __name__ == "__main__":
    test_cases = [
        (0, 0),
        (1, 5),
        (10, 20),
        (50, 100),
        (100, 200)
    ]

    for resume_idx, jd_idx in test_cases:
        print("=" * 60)
        print(f"Resume {resume_idx} vs JD {jd_idx}")

        result = match_resume_to_jd(
            resume_idx,
            jd_idx
        )

        for key, value in result.items():
            print(f"{key}: {value}")