"""
Person 3 — Part 3: Pace / Pause Communication Metrics
========================================================
Workaround for the filler-word limitation documented in the STT report:
instead of counting filler words from the transcript, we measure delivery
quality directly from the audio signal.

- words-per-minute (WPM): transcript word count / audio duration
- pause detection: silence gaps found via librosa energy analysis
"""

import librosa
import numpy as np


def analyze_pace_and_pauses(
    audio_path: str,
    transcript: str,
    top_db: int = 30,
    min_pause_sec: float = 0.3,
):
    """
    Compute pace (WPM) and pause metrics for a single audio file + its transcript.

    Parameters
    ----------
    audio_path : str
        Path to the .wav file (e.g. data/raw/audio_samples/Actor_01/....wav)
    transcript : str
        The transcript text for this audio (passed in directly — caller decides
        where it comes from: Whisper output, a CSV column, a .txt file, etc.)
    top_db : int
        Threshold (in dB below peak) below which audio is considered silence.
        Lower = stricter (more audio counted as silence). 30 is a reasonable
        default for speech; tune this against a few known samples if pause
        counts look off.
    min_pause_sec : float
        Minimum gap length (seconds) to count as a real pause, filtering out
        tiny sub-phoneme gaps that aren't meaningful pauses.

    Returns
    -------
    dict with:
        duration_sec, word_count, wpm,
        pause_count, total_pause_sec, avg_pause_sec, longest_pause_sec,
        pause_ratio (fraction of total duration spent paused)
    """
    # --- Load audio ---
    y, sr = librosa.load(audio_path, sr=None)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    # --- Pace: words per minute ---
    word_count = len(transcript.split())
    wpm = (word_count / (duration_sec / 60.0)) if duration_sec > 0 else 0.0

    # --- Pause detection via energy-based silence splitting ---
    # librosa.effects.split returns [start, end] sample indices of
    # NON-silent intervals. Gaps BETWEEN consecutive intervals are pauses.
    intervals = librosa.effects.split(y, top_db=top_db)

    pause_durations = []
    if len(intervals) > 1:
        for i in range(len(intervals) - 1):
            gap_start_sample = intervals[i][1]
            gap_end_sample = intervals[i + 1][0]
            gap_sec = (gap_end_sample - gap_start_sample) / sr
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
# Quick manual test (run this file directly to sanity-check on one sample)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    def _find_project_root(marker="data/processed/audio/train.csv", max_up=4):
        p = Path.cwd()
        for _ in range(max_up):
            if (p / marker).exists():
                return p
            p = p.parent
        raise FileNotFoundError(f"Could not locate '{marker}' from {Path.cwd()}")

    root = _find_project_root()
    # Auto-discover any real .wav file under data/raw/audio_samples/ to test with
    audio_dir = root / "data/raw/audio_samples"
    candidates = sorted(audio_dir.glob("**/*.wav"))
    sample_transcript = "This is a placeholder transcript for a quick manual test."

    if not candidates:
        print(f"No .wav files found under {audio_dir} — check the folder path.")
        sys.exit(0)

    sample_audio = candidates[0]
    print(f"Testing with: {sample_audio}")

    metrics = analyze_pace_and_pauses(str(sample_audio), sample_transcript)
    print(metrics)