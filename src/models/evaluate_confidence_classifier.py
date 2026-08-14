"""
Person 3 — Part 2: Evaluate the Audio Confidence Classifier
==============================================================
Loads the model + label encoder saved in Part 1 and evaluates on
data/processed/audio/val.csv and data/processed/audio/test.csv.
Reports accuracy, per-class precision/recall/F1, and confusion matrix.

Run this from the project root or from inside notebooks/ — path is
auto-detected the same way as Part 1.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# 1. Locate project root (same approach as Part 1)
# ---------------------------------------------------------------------------
def _find_project_root(marker="data/processed/audio/train.csv", max_up=4):
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
DATA_DIR = PROJECT_ROOT / "data/processed/audio"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "confidence_classifier.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "confidence_label_encoder.joblib"

FEATURE_COLS = [f"mfcc_{i}" for i in range(13)] + [
    "pitch_mean",
    "energy_mean",
    "zcr_mean",
    "duration",
]
TARGET_COL = "confidence_label"

# ---------------------------------------------------------------------------
# 2. Load model + label encoder (from Part 1)
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No model found at {MODEL_PATH}. Run Part 1 (train_confidence_classifier.py) first."
    )

clf = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
print(f"Loaded model from {MODEL_PATH}")
print(f"Classes: {list(label_encoder.classes_)}\n")


# ---------------------------------------------------------------------------
# 3. Helper: evaluate on one split
# ---------------------------------------------------------------------------
def evaluate_split(csv_path: Path, split_name: str):
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {missing}")

    X = df[FEATURE_COLS]
    y_true_raw = df[TARGET_COL]

    # Guard against unseen labels in val/test that weren't in train
    unseen = set(y_true_raw.unique()) - set(label_encoder.classes_)
    if unseen:
        raise ValueError(
            f"{split_name} contains labels never seen in train.csv: {unseen}"
        )

    y_true = label_encoder.transform(y_true_raw)
    y_pred = clf.predict(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"=== {split_name} ({len(df)} samples) ===")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(
        classification_report(
            y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
        )
    )
    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(cm_df)
    print()
    return acc


# ---------------------------------------------------------------------------
# 4. Run on val.csv and test.csv
# ---------------------------------------------------------------------------
val_acc = evaluate_split(DATA_DIR / "val.csv", "VALIDATION")
test_acc = evaluate_split(DATA_DIR / "test.csv", "TEST")

print("=== Summary ===")
print(f"Val accuracy:  {val_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")