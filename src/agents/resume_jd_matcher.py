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

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    base_path = os.path.join("data", "processed", "text")

    resume_df = pd.read_csv(os.path.join(base_path, "unified_resumes.csv"))
    jd_df = pd.read_csv(os.path.join(base_path, "unified_job_descriptions.csv"))

    return resume_df, jd_df

def match_resume_to_jd(resume_index, jd_index):

    resume_df, jd_df = load_data()

    resume_text = resume_df.iloc[resume_index]["raw_text"]
    jd_text = jd_df.iloc[jd_index]["description"]

    # Similarity score
    resume_embedding = model.encode(resume_text)
    jd_embedding = model.encode(jd_text)

    score = cosine_similarity(
        [resume_embedding],
        [jd_embedding]
    )[0][0] * 100

    # Resume skills
    resume_skills = ast.literal_eval(resume_df.iloc[resume_index]["skills"])
    resume_lower = [s.lower() for s in resume_skills]

    # Extract skills from JD
    jd_skills = []

    for skill in SKILLS_LIST:
        if skill.lower() in jd_text.lower():
            jd_skills.append(skill)

    matched = [s for s in jd_skills if s.lower() in resume_lower]
    missing = [s for s in jd_skills if s.lower() not in resume_lower]

    return {
        "Resume ID": resume_df.iloc[resume_index]["resume_id"],
        "Job Title": jd_df.iloc[jd_index]["job_title"],
        "Match Score": float(round(score, 2)),
        "Matched Skills": matched,
        "Missing Skills": missing
    }

if __name__ == "__main__":
    result = match_resume_to_jd(0, 0)
    print(result)

if __name__ == "__main__":
    result = match_resume_to_jd(0, 0)

    print("Resume-JD Matching Result")
    print("-------------------------")
    print(result)