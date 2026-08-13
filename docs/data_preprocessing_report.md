# Data Sources & Preprocessing Report — AI-Powered-Voice-Based-Mock-Interview-Coach

This document records every dataset used in the project, where it came from, its original size, what cleaning was applied, and the key decisions made along the way. It doubles as the "Data Collection & Preprocessing" section for the project report.

---

## 1. Overview

The system takes a user's **resume + target job description** and conducts a **speech-based mock interview**, asking role-relevant HR and technical questions, then scoring both answer quality and vocal confidence. This required three categories of data:

| Category | Purpose |
|---|---|
| Interview Q&A (HR + Technical) | Question bank + ideal-answer grounding for the Evaluation Agent |
| Resumes & Job Descriptions | Training/reference data for resume↔JD skill matching |
| Emotional speech audio | Confidence/tone signal for the Speech/Text Agent |

---

## 2. Text Datasets

### 2.1 HR Interview Questions and Ideal Answers
- **Source:** Kaggle — `aryan208/hr-interview-questions-and-ideal-answers`
- **Downloaded:** [fill in date you ran the download]
- **Raw size:** 2,500,000 rows × 8 columns (`question, category, role, experience, difficulty, source_type, ideal_answer, keywords`)
- **Data quality finding:** 2,465,440 rows (98.6%) were exact duplicates — the dataset was generated from a small template set (8 categories × 8 roles × 3 difficulty levels) padded to appear large.
- **Action taken:** Dropped duplicates on all fields except `keywords` (list-typed, unhashable) → **34,760 unique rows** retained.
- **License:** Kaggle dataset, free for academic/educational use (verify attribution requirement on dataset page before public release).

### 2.2 Software Engineering Interview Questions Dataset
- **Source:** Kaggle — `syedmharis/software-engineering-interview-questions-dataset`
- **Raw size:** 200 rows × 5 columns (`Question Number, Question, Answer, Category, Difficulty`)
- **Data quality finding:** File required `encoding="latin1"` to load (not valid UTF-8). Zero duplicates, zero nulls.
- **Known limitation:** Only 200 rows — thin coverage for an "industrial" technical interview scope (DSA/System Design/ML). Flagged for future expansion with additional technical Q&A sources.

### 2.3 Unified Question Bank
- **Built from:** 2.1 + 2.2, standardized to a common schema (`question, answer, category, role, difficulty, domain`)
- **Final size:** 34,960 rows (34,760 HR + 200 technical)
- **Output files:** `data/processed/text/unified_questions.csv`, plus stratified `train.csv` (24,332) / `val.csv` (5,214) / `test.csv` (5,214)
- **Stratification note:** Split on a `category_grouped` column where categories with fewer than 10 total occurrences were merged into `"Other"`, since several technical categories had only 1–2 members and broke stratified splitting otherwise.
- **Embeddings:** Sentence embeddings generated for `train.csv` questions using `sentence-transformers/all-MiniLM-L6-v2` (384-dim) → `train_question_embeddings.npy`

### 2.4 Resume Datasets (3 sources merged)

| Source | Type | Raw size | Notes |
|---|---|---|---|
| `resume_data.csv` (Kaggle) | Structured fields (35 cols): career objective, skills, education, work history, matched JD requirements + match score | 9,544 rows | 56 rows missing `skills`, dropped → 9,488 kept |
| `train.json` (Kaggle, dataturks NER) | Raw resume text + character-span entity annotations (`SKILL, EDUCATION, COMPANY`, etc.) | 5,960 rows | Required custom parsing — file had malformed unicode surrogate pairs, cleaned via regex before JSON parsing; skill spans extracted and split on delimiters |
| `Resume.csv` (Kaggle) | Plain resume text + 24 job categories | 2,484 rows | Simplest structure, used as-is (`Resume_str`, `Category`) |

- **Unified output:** `unified_resumes.csv` — 17,932 rows, common schema (`resume_id, source, raw_text, skills, category`)
- **Cleaning applied to all 3:** lowercased, HTML/URL/special-character stripped, tokenized, lemmatized (NLTK)

### 2.5 Job Description Datasets (3 sources merged)

| Source | Raw size | Notes |
|---|---|---|
| `data.csv` (Kaggle) | 521 rows (`Job Title, Description`) | Clean, no nulls/duplicates |
| `job_title_des.csv` (Kaggle) | 2,277 rows (`Job Title, Job Description` + stray index column) | Index column dropped before merge |
| `training_data.csv` (Kaggle) | 853 rows | Contained a nested JSON string (`model_response`) with LLM-parsed fields (`Required Skills, Experience Level, Preferred Qualifications`, etc.) — parsed with `json_normalize` |

- **Unified output:** `unified_job_descriptions.csv` — 3,651 rows, common schema (`job_title, description, source, required_skills, experience_level`)

---

## 3. Audio Dataset

### 3.1 RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Source:** Kaggle — `uwrfkaggler/ravdess-emotional-speech-audio`
- **Raw size:** 1,440 `.wav` clips across 24 actors, 8 emotion classes, 48kHz sample rate
- **Emotion distribution:** calm/happy/sad/disgust/angry/fearful/surprised = 192 each, neutral = 96 (no "strong intensity" variant)
- **Labels parsed from filename convention** (e.g. `03-01-06-01-02-01-12.wav` → modality-channel-emotion-intensity-statement-repetition-actor)
- **Confidence mapping applied:**
  - `calm, neutral, happy` → **confident** (480 clips)
  - `sad, fearful, disgust, angry` → **nervous** (768 clips)
  - `surprised` → **neutral** (192 clips, kept ambiguous rather than forced into either bucket)

### 3.2 Audio Preprocessing Pipeline
For each clip:
1. Resampled to 16kHz mono (standard for speech models)
2. Leading/trailing silence trimmed (`librosa.effects.trim`, top_db=20)
3. Features extracted: 13 MFCCs (mean-pooled), pitch (mean, via `piptrack`), energy (RMS mean), zero-crossing rate (speaking-rate proxy), clip duration
4. All numeric features standardized (`StandardScaler`, mean≈0, std≈1)

- **Output:** `audio_features_raw.csv` (1,440 rows × 21 columns), plus stratified `train.csv` (1,008) / `val.csv` (216) / `test.csv` (216), split on `confidence_label`

### 3.3 Known limitation
RAVDESS uses **scripted, acted emotional speech**, not real interview responses — it's a reasonable proxy for training a confidence/tone classifier but won't perfectly reflect real interview nervousness patterns. Noted as a limitation for the project report; a natural extension would be fine-tuning on real interview recordings (e.g., MIT Interview Dataset) if access becomes available.

---

## 4. Summary Table

| Dataset | Raw Rows | Final Rows | Output Location |
|---|---|---|---|
| HR Questions | 2,500,000 | 34,760 | `data/processed/text/unified_questions.csv` |
| Technical Questions | 200 | 200 | (merged above) |
| Resumes (3 sources) | 17,988 | 17,932 | `data/processed/text/unified_resumes.csv` |
| Job Descriptions (3 sources) | 3,651 | 3,651 | `data/processed/text/unified_job_descriptions.csv` |
| RAVDESS Audio | 1,440 | 1,440 | `data/processed/audio/audio_features_raw.csv` |

**Total processing outputs:** 3 unified text tables + train/val/test splits, sentence embeddings, 1 unified audio feature table + train/val/test splits.

---

## 5. Tools & Libraries Used
- `pandas`, `numpy` — data handling
- `nltk` — tokenization, lemmatization, stopword removal
- `scikit-learn` — stratified train/val/test splitting, feature scaling
- `sentence-transformers` (`all-MiniLM-L6-v2`) — text embeddings
- `librosa`, `soundfile` — audio loading and feature extraction
- `kaggle` CLI — dataset acquisition

---

