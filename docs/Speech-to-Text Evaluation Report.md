# Speech-to-Text Evaluation Report — AI-Powered Voice-Based Mock Interview Coach

## 1. Objective
Evaluate the accuracy and real-world reliability of the Speech-to-Text (STT) component before integrating it into the Speech/Text Agent, and determine whether it meets the project's 93%+ accuracy target.

## 2. Model Used
- **Model:** Whisper (OpenAI) — transformer-based automatic speech recognition (ASR)
- **Variant:** `small` (also tested `base` for comparison)
- **Library:** `openai-whisper`
- **Why Whisper:** industry-standard, open-source, pretrained on 680,000+ hours of multilingual speech — no training from scratch required (not feasible for a project of this scope); only evaluation/testing was performed.

## 3. Evaluation Methodology
Speech-to-text accuracy is measured using **Word Error Rate (WER)**, converted to accuracy as `(1 − WER) × 100`. WER compares a model's transcription against a known correct ("ground truth") transcript, word by word.

Two test phases were run:

### Phase 1 — Controlled baseline (RAVDESS dataset)
- 1,440 short (~3 sec) scripted speech clips, each one of two fixed sentences ("Kids are talking by the door" / "Dogs are sitting by the door")
- Ground truth derived directly from RAVDESS's filename encoding (known sentence per clip)
- Tested on samples of 50 and 200 clips

### Phase 2 — Real, spontaneous speech test
- 10 self-recorded answers to real interview questions (HR/behavioral), delivered naturally and unscripted (not read from a script) to simulate genuine interview conditions
- Ground truth manually transcribed by listening back to each recording, written down verbatim including filler words ("um", "uh") and self-corrections

## 4. Results

| Test | Sample Size | Scoring Method | Accuracy |
|---|---|---|---|
| RAVDESS (raw comparison) | 50 clips | Verbatim (unnormalized) | 83.3% |
| RAVDESS (punctuation-normalized) | 50 clips | Verbatim, punctuation removed | 99.3% |
| RAVDESS (punctuation-normalized) | 200 clips | Verbatim, punctuation removed | 99.6% |
| Real speech | 10 recordings | Verbatim (including fillers) | 80.3% |
| Real speech | 10 recordings | **Content-only** (fillers/stutters excluded) | **95.5%** |

## 5. Key Finding: Verbatim vs. Content Accuracy
Initial real-speech testing showed only 80.3% accuracy — well below target. Manual inspection of every transcription revealed the cause: **Whisper was transcribing the actual content of every answer correctly**, but consistently omitted filler words ("um", "uh") and smoothed over self-corrections/stutters. This is expected, intentional behavior — Whisper is trained to output clean, readable text rather than a verbatim court-reporter transcript. It is not mishearing words; it is not transcribing disfluencies by design.

Re-scoring with filler words and stutter artifacts excluded from both the ground truth and the prediction (content-only comparison) raised measured accuracy to **95.5%**, confirming that Whisper's true content-transcription accuracy exceeds the project's 93% target. This finding was also independently confirmed by the project supervisor as an expected model characteristic rather than a defect.

## 6. Product Implication (Scoped Out for Now)
Because Whisper does not reliably preserve filler words in its output, **transcript-based filler-word counting is not currently a reliable method** for the "Communication Skills" feedback feature. This is a known, documented limitation.

**Decision:** This limitation is being scoped out of the current build phase to keep the project on schedule. It is planned as a **future enhancement** — a dedicated audio-level disfluency/pause detection step (independent of the Whisper transcript) would be required to support precise filler-word counting. In the interim, communication feedback can rely on pace (words per minute) and pause-length metrics derived directly from the audio signal, which do not depend on Whisper capturing filler words.

## 7. Conclusion
The Whisper (`small`) STT component meets the project's accuracy target (95.5% content accuracy on real, spontaneous speech vs. 93% goal) and is approved for integration into the Speech/Text Agent. The single known limitation (verbatim filler-word transcription) has a clear, low-risk workaround and is deferred to future work rather than blocking current development.