"""
Person 3 — Part 1: Train the Audio Confidence Classifier
==========================================================
Trains a Random Forest on data/processed/audio/train.csv to predict
`confidence_label` from extracted audio features (MFCCs, pitch, energy, ZCR).

Run this from the project root.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
def _find_project_root(marker="data/processed/audio/train.csv", max_up=4):
    """Walk up from cwd until we find the expected data file, so this
    works whether run from project root or from notebooks/."""
    p = Path.cwd()
    for _ in range(max_up):
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError(
        f"Could not locate '{marker}' by walking up from {Path.cwd()}. "
        "Check that you're inside the project folder."
    )


PROJECT_ROOT = _find_project_root()
TRAIN_PATH = PROJECT_ROOT / "data/processed/audio/train.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "confidence_classifier.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "confidence_label_encoder.joblib"

# Feature columns: 13 MFCCs + pitch/energy/zcr means + duration
FEATURE_COLS = [f"mfcc_{i}" for i in range(13)] + [
    "pitch_mean",
    "energy_mean",
    "zcr_mean",
    "duration",
]
TARGET_COL = "confidence_label"

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(TRAIN_PATH)
print(f"Loaded train.csv: {df.shape}")
print(f"Label distribution:\n{df[TARGET_COL].value_counts()}\n")

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected feature columns: {missing}")

X_train = df[FEATURE_COLS]
y_train_raw = df[TARGET_COL]

# ---------------------------------------------------------------------------
# 3. Encode labels (confident / nervous / neutral -> 0/1/2)
# ---------------------------------------------------------------------------
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
print(f"Label classes: {list(label_encoder.classes_)}")

# ---------------------------------------------------------------------------
# 4. Train Random Forest
#    class_weight='balanced' because labels are imbalanced
#    (nervous:538, confident:336, neutral:134 in train.csv)
# ---------------------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
print(f"\nTrain accuracy (on training set itself, NOT a real metric): {train_acc:.4f}")
print("Real evaluation happens on val.csv/test.csv in Part 2 — not here.")

# ---------------------------------------------------------------------------
# 5. Feature importances (sanity check)
# ---------------------------------------------------------------------------
importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(
    ascending=False
)
print("\nTop 5 most important features:")
print(importances.head(5))

# ---------------------------------------------------------------------------
# 6. Save model + label encoder
# ---------------------------------------------------------------------------
joblib.dump(clf, MODEL_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)
print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved label encoder to: {LABEL_ENCODER_PATH}")