"""
Person 3 — Part 4b: analyze_delivery(audio_file, transcript)
================================================================
Combines everything from Parts 1-3 into the single deliverable function:

    analyze_delivery(audio_file, transcript) -> confidence score + pace + pause metrics

Requires (run these once first, in order, if not already done):
  1. train_confidence_classifier.py   -> models/confidence_classifier.joblib
                                          models/confidence_label_encoder.joblib
  2. fit_feature_scaler.py            -> models/audio_feature_scaler.joblib

Feature extraction below is copied EXACTLY from notebook 02's
extract_audio_features() (TARGET_SR=16000, top_db=20 trim, n_mfcc=13,
piptrack pitch, RMS energy, ZCR) so it matches how train.csv was built.
"""
from src.agents.pace_pause_metrics import analyze_pace_and_pauses
import librosa
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ---------------------------------------------------------------------------
# 0. Locate project root + load saved artifacts
# ---------------------------------------------------------------------------
def _find_project_root(marker="data/processed/audio/train.csv", max_up=4):
    p = Path.cwd()
    for _ in range(max_up):
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError(f"Could not locate '{marker}' from {Path.cwd()}")


PROJECT_ROOT = _find_project_root()
MODEL_DIR = PROJECT_ROOT / "models"

CLASSIFIER_PATH = MODEL_DIR / "confidence_classifier.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "confidence_label_encoder.joblib"
SCALER_PATH = MODEL_DIR / "audio_feature_scaler.joblib"

FEATURE_COLS = [f"mfcc_{i}" for i in range(13)] + [
    "pitch_mean",
    "energy_mean",
    "zcr_mean",
    "duration",
]

TARGET_SR = 16000  # must match notebook 02


def _load_artifacts():
    for path, label in [
        (CLASSIFIER_PATH, "classifier (run train_confidence_classifier.py)"),
        (LABEL_ENCODER_PATH, "label encoder (run train_confidence_classifier.py)"),
        (SCALER_PATH, "feature scaler (run fit_feature_scaler.py)"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} at {path}")
    clf = joblib.load(CLASSIFIER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, label_encoder, scaler


_CLF, _LABEL_ENCODER, _SCALER = _load_artifacts()


# ---------------------------------------------------------------------------
# 1. Raw feature extraction — copied exactly from notebook 02
# ---------------------------------------------------------------------------
def extract_audio_features(file_path: str):
    """Same logic as notebook 02's extract_audio_features(). Returns a dict
    of RAW (unscaled) features, or None if the file can't be processed."""
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)

        if len(audio_trimmed) == 0:
            return None

        mfccs = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        pitches, magnitudes = librosa.piptrack(y=audio_trimmed, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0

        rms = librosa.feature.rms(y=audio_trimmed)
        energy_mean = np.mean(rms)

        zcr = librosa.feature.zero_crossing_rate(audio_trimmed)
        zcr_mean = np.mean(zcr)

        duration = len(audio_trimmed) / sr

        return {
            **{f"mfcc_{i}": mfccs_mean[i] for i in range(13)},
            "pitch_mean": pitch_mean,
            "energy_mean": energy_mean,
            "zcr_mean": zcr_mean,
            "duration": duration,
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# 2. Pace / pause metrics — from Part 3 (pace_pause_metrics.py)
#    NOTE: uses sr=None (original sample rate) here, matching Part 3 as
#    already validated — separate from the TARGET_SR=16000 load above,
#    which is only for the classifier's MFCC/pitch/energy/ZCR features.
# ---------------------------------------------------------------------------
def analyze_pace_and_pauses(
    audio_path: str,
    transcript: str,
    top_db: int = 30,
    min_pause_sec: float = 0.3,
):
    y, sr = librosa.load(audio_path, sr=None)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    word_count = len(transcript.split())
    wpm = (word_count / (duration_sec / 60.0)) if duration_sec > 0 else 0.0

    intervals = librosa.effects.split(y, top_db=top_db)
    pause_durations = []
    if len(intervals) > 1:
        for i in range(len(intervals) - 1):
            gap_sec = (intervals[i + 1][0] - intervals[i][1]) / sr
            if gap_sec >= min_pause_sec:
                pause_durations.append(gap_sec)

    pause_count = len(pause_durations)
    total_pause_sec = float(sum(pause_durations))
    avg_pause_sec = float(np.mean(pause_durations)) if pause_durations else 0.0
    longest_pause_sec = float(max(pause_durations)) if pause_durations else 0.0
    pause_ratio = (total_pause_sec / duration_sec) if duration_sec > 0 else 0.0

    return {
        "duration_sec": round(duration_sec, 3),
        "word_count": word_count,
        "wpm": round(wpm, 2),
        "pause_count": pause_count,
        "total_pause_sec": round(total_pause_sec, 3),
        "avg_pause_sec": round(avg_pause_sec, 3),
        "longest_pause_sec": round(longest_pause_sec, 3),
        "pause_ratio": round(pause_ratio, 3),
    }


# ---------------------------------------------------------------------------
# 3. THE DELIVERABLE: analyze_delivery()
# ---------------------------------------------------------------------------
def analyze_delivery(audio_file: str, transcript: str):
    """
    Full delivery analysis for one audio file + its transcript.

    Parameters
    ----------
    audio_file : str
        Path to a .wav file.
    transcript : str
        The transcript text for this audio (caller supplies it — from
        Whisper output, a CSV column, wherever).

    Returns
    -------
    dict with:
        confidence_label   : predicted class ('confident' / 'nervous' / 'neutral')
        confidence_scores  : dict of class -> predicted probability
        pace_pause_metrics : dict from analyze_pace_and_pauses()
    """
    # --- Confidence classification ---
    raw_features = extract_audio_features(audio_file)
    if raw_features is None:
        raise ValueError(f"Could not extract features from {audio_file}")

    features_df = pd.DataFrame([raw_features])[FEATURE_COLS]
    features_scaled = pd.DataFrame(
        _SCALER.transform(features_df), columns=FEATURE_COLS
    )

    pred_idx = _CLF.predict(features_scaled)[0]
    pred_label = _LABEL_ENCODER.inverse_transform([pred_idx])[0]

    proba = _CLF.predict_proba(features_scaled)[0]
    confidence_scores = {
        cls: round(float(p), 4) for cls, p in zip(_LABEL_ENCODER.classes_, proba)
    }

    # --- Pace / pause metrics ---
    pace_pause = analyze_pace_and_pauses(audio_file, transcript)

    return {
        "confidence_label": pred_label,
        "confidence_scores": confidence_scores,
        "pace_pause_metrics": pace_pause,
    }


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    audio_dir = PROJECT_ROOT / "data/raw/audio_samples"
    candidates = sorted(audio_dir.glob("**/*.wav"))
    if not candidates:
        print(f"No .wav files found under {audio_dir}")
    else:
        sample_audio = str(candidates[0])
        sample_transcript = "This is a placeholder transcript for a quick manual test."
        print(f"Testing with: {sample_audio}\n")
        result = analyze_delivery(sample_audio, sample_transcript)
        print(result)