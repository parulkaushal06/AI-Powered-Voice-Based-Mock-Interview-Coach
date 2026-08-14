"""
Person 3 — Part 4a: Fit and Save the Feature Scaler
======================================================
The original scaler used to produce train/val/test.csv (in notebook 02,
cell 11) was never saved. This refits an identical StandardScaler from
data/processed/audio/audio_features_raw.csv — verified to reproduce
train.csv's scaled values exactly (max abs diff ~1e-7, floating point noise).

Run this ONCE. It saves models/audio_feature_scaler.joblib, which
analyze_delivery() (Part 4b) loads to scale features from new audio files.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def _find_project_root(marker="data/processed/audio/train.csv", max_up=4):
    p = Path.cwd()
    for _ in range(max_up):
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError(f"Could not locate '{marker}' from {Path.cwd()}")


PROJECT_ROOT = _find_project_root()
RAW_FEATURES_PATH = PROJECT_ROOT / "data/processed/audio/audio_features_raw.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
SCALER_PATH = MODEL_DIR / "audio_feature_scaler.joblib"

FEATURE_COLS = [f"mfcc_{i}" for i in range(13)] + [
    "pitch_mean",
    "energy_mean",
    "zcr_mean",
    "duration",
]

raw_df = pd.read_csv(RAW_FEATURES_PATH)
print(f"Loaded raw features: {raw_df.shape}")

missing = [c for c in FEATURE_COLS if c not in raw_df.columns]
if missing:
    raise ValueError(f"audio_features_raw.csv missing expected columns: {missing}")

scaler = StandardScaler()
scaler.fit(raw_df[FEATURE_COLS])

joblib.dump(scaler, SCALER_PATH)
print(f"Saved scaler to: {SCALER_PATH}")
print("\nSanity check — scaler mean per feature:")
print(pd.Series(scaler.mean_, index=FEATURE_COLS))