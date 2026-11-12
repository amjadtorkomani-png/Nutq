# Nutq Listener · سامع النطق: R model (v0.1)

The first model of the Nutq "listener": it hears one word said by a child and decides whether the **R (ر)** in that word is said correctly.

> **Important:** this model is trained and tested on **synthetic speech only** (text-to-speech voices turned into child-like voices). Its numbers show the method works on that data. They say **nothing yet** about real children. The next milestone is a test on real recordings labelled by speech specialists.

This folder has three parts:

| Part | Where | What |
|---|---|---|
| **1. The model** | [`trained/r_baseline.joblib`](trained/r_baseline.joblib) + [`scripts/`](scripts/) | The trained model, and all the code to rebuild it and use it |
| **2. The data** | [`data/`](data/) | Word list, every generated utterance, the split, metadata for all 74,336 clips, and audio samples |
| **3. The results** | [`reports/`](reports/) and the [Results](#4-results) section below | Full evaluation |

---

## Contents

1. [Quick start: use the model](#1-quick-start-use-the-model)
2. [What the model does](#2-what-the-model-does)
3. [The data](#3-the-data)
4. [Results](#4-results)
5. [Known problem: five voices](#5-known-problem-five-voices)
6. [The model and its features](#6-the-model-and-its-features)
7. [Rebuild everything](#7-rebuild-everything)
8. [Files](#8-files)
9. [Next steps](#9-next-steps)

---

## 1. Quick start: use the model

```bash
pip install -r requirements.txt
python scripts/predict.py data/samples/w01_رمل_r.wav data/samples/w19_قمر_gh.wav
```

Output on the samples in this folder:

| File | What was said | P(error) | Verdict |
|---|---|---|---|
| `w01_رمل_r.wav` | رَمْل (correct) | 0.00 | correct |
| `w01_رمل_y.wav` | يَمْل (R → Y) | 1.00 | wrong |
| `w01_رمل_mix-y-35.wav` | R blended 35% towards Y | 0.44 | close → specialist |
| `w01_رمل_mix-y-65.wav` | R blended 65% towards Y | 0.92 | wrong |
| `w12_كرة_r.wav` | كُرَة (correct) | 0.00 | correct |
| `w12_كرة_l.wav` | كُلَة (R → L) | 0.82 | close → specialist |
| `w12_كرة_mix-l-35.wav` | R blended 35% towards L | 0.00 | correct |
| `w12_كرة_mix-l-65.wav` | R blended 65% towards L | 0.18 | close → specialist |
| `w19_قمر_r.wav` | قَمَر (correct) | 0.00 | correct |
| `w19_قمر_gh.wav` | قَمَغ (R → GH) | 0.93 | wrong |
| `w19_قمر_y.wav` | قَمَي (R → Y) | 1.00 | wrong |
| `w19_قمر_mix-gh-65.wav` | R blended 65% towards GH | 0.32 | close → specialist |

The verdict uses a **confidence gate**: `correct` if P(error) ≤ 0.1, `wrong` if ≥ 0.9, otherwise `close`, and in the app the clip goes to the specialist.

**Input:** WAV, one word, mono (other sample rates are resampled to 16 kHz).

---

## 2. What the model does

| | |
|---|---|
| **Input** | Audio of one word. Nothing else: the model is not told which word was asked. |
| **Output** | Probability that the R is wrong, calibrated, plus a verdict through the confidence gate |
| **Target sound** | R (ر) |
| **Errors it learned** | R replaced by 19 other letters (غ ل ي و ب د ن ت ع ج م ه خ ق ك ز ذ ط ح), or R dropped |
| **Speakers** | Child-like voices (ages 5–9 target), Arabic dialects from 15 countries |

Why not speech-to-text? A child says **غَمْل** instead of **رَمْل**, and speech-to-text writes "رمل" and marks it correct. The listener judges the **sound**, not the word.

---

## 3. The data

No recordings of real children are used or stored. All training data is generated.

### 3.1 How it was made

```
25 words  ──►  497 utterances  ──►  TTS (edge-tts)  ──►  in-between R  ──►  child-like voice  ──►  automatic check
              (correct + errors)    44 voices × 2 speeds   (WORLD vocoder)    (Praat, 4 per voice)
```

**Step 1: words.** [`data/words_r.csv`](data/words_r.csv) has 25 everyday words a child knows, with R at the start (9), middle (9) and end (7) of the word: رمل، رجل، رأس، رمان، ريشة، رقم، رسم، روبوت، ربيع، أرنب، برتقال، كرة، شجرة، سيارة، طائرة، بقرة، فراشة، قرد، قمر، بحر، نمر، تمر، حجر، نهر، صقر.

Rules for the list: one R per word, no غ in the word, and no two words that differ only in the R (so that رول/غول can't both be in the list).

**Step 2: utterances.** [`data/utterances_r.csv`](data/utterances_r.csv), made by `build_utterances.py`. For every word: the correct form, plus R replaced by each of 19 letters, plus R deleted. A variant is dropped if it can't be pronounced (كُرَة without R → "كُة") or if it doubles a neighbour letter. Result: **497 utterances (25 correct, 472 errors)**, all with full diacritics.

TTS spelling fix: edge TTS dropped the final vowel of words ending in ة (فراشة was read "فراش", كرة "كُر"). The final ة is written as ه for the TTS only (كُرَه), checked by ear.

**Step 3: adult TTS voices.** `generate_tts.py`, using edge-tts.
- 32 Arabic voices from 16 countries, plus 12 "Multilingual" foreign voices, each at normal speed and 20% slower.
- Per word and voice: the correct form plus 4 sampled errors (common errors ل غ ي و and deletion are picked twice as often).
- **11,085 adult clips**, 16 kHz mono.

**Step 4: in-between R sounds.** `make_blends.py`. Real children often say a sound *between* R and another letter. For one voice and word, the correct clip and an error clip (ل غ ي و) are aligned with DTW, the region where they differ (the R) is found, and inside it the WORLD vocoder spectral envelope is mixed at 35% and 65%. Both originals are also passed through the vocoder, so vocoder artefacts appear in every class. **7,499 clips.**

**Step 5: child-like voices.** `childify.py`, using Praat "Change gender". Each source voice becomes 4 "virtual children" with their own fixed settings:

| Setting | Range | Why |
|---|---|---|
| Formant shift | ×1.12 to ×1.28 | Shorter vocal tract |
| Pitch (F0) | 230–320 Hz | Children's voice range |
| Pitch range | ×0.8 to ×1.4 | Flat or lively intonation |
| Speed | ×1.00 to ×1.25 | Children often speak slower |

Checked by measurement: the measured spectral stretch matches the setting (correlation 0.99). **74,336 child-like clips.**

**Step 6: automatic check.** `check_child.py` reads every clip: broken file, too short, no voiced part, or F0 outside 180–400 Hz. 170 clips flagged and excluded ([`data/excluded_clips.txt`](data/excluded_clips.txt)).

### 3.2 Checked by ear

| Problem found | Fix |
|---|---|
| ة not pronounced at word end | Final ة written as ه for the TTS |
| 3 voices (ar-AE-Fatima, ar-LB-Layla, ar-SA-Zariyah) said "ارمل" for رمل and "عجل" for رجل, already in the adult TTS | Voices excluded ([`data/excluded_voices.txt`](data/excluded_voices.txt)) |
| Free text-to-speech model MMS sounded poor | Removed |
| Strongest formant shift (×1.32) sounded damaged | Maximum lowered to ×1.28 |

### 3.3 The split

[`data/splits.csv.gz`](data/splits.csv.gz), made by `make_splits.py`. A voice is never on both sides: all its speeds and virtual children stay together.

| Set | Clips | Content | Used for |
|---|---|---|---|
| `train` | 32,834 | Correct + errors (incl. vocoder copies), 23 Arabic voices, 5 folds by voice | Training and cross-validation |
| `test_voices` | 8,503 | Same kind, 6 Arabic voices (3 male, 3 female) never seen in training | Final test |
| `confidence` | 14,931 | In-between R sounds | **Not trained on.** Tests that confidence drops as R is blended |
| `held_foreign` | 11,975 | Foreign voices | Held out: their "correct" R is not verified, and training on their errors only would teach "foreign accent = error" |
| `excluded` | 6,093 | Excluded voices and clips | Not used |

Training labels: 8,374 correct and 24,460 errors (class weights balance them).

### 3.4 What is in this repository

| File | Rows | Content |
|---|---|---|
| `data/words_r.csv` | 25 | Word list |
| `data/utterances_r.csv` | 497 | Every utterance with label, replaced letter and TTS text |
| `data/splits.csv.gz` | 74,336 | Set and fold of every clip |
| `data/metadata/raw_tts.csv` | 11,085 | Adult TTS clips: voice, speed, text, label, duration, loudness |
| `data/metadata/blends.csv` | 7,499 | In-between clips: blend amount and the detected R region |
| `data/metadata/child.csv.gz` | 74,336 | Child-like clips: source clip, formant / pitch / speed settings |
| `data/metadata/child_check.csv` | 170 | Clips flagged by the automatic check |
| `data/samples/` | 12 | Example WAV files (see `samples.csv`) |

**Not in the repository:** the 1.5 GB of audio and the 57 MB feature file. Both are rebuilt by the scripts ([section 7](#7-rebuild-everything)).

---

## 4. Results

All numbers below are on synthetic data. Full report: [`reports/baseline_report.md`](reports/baseline_report.md); all numbers: [`reports/baseline_metrics.json`](reports/baseline_metrics.json).

"Error" is the positive class. **False rejection** = a correct word marked wrong (bad for the child). **Missed errors** = a wrong R marked correct.

### 4.1 Overall

| Measure | Cross-validation (by voice) | **Test: 6 unseen voices** | Baseline (logistic regression), test |
|---|---|---|---|
| Balanced accuracy | 91.0% | **93.1%** | 84.5% |
| Errors caught (recall) | 98.7% | **98.7%** | 87.5% |
| False rejection | 16.6% | **12.6%** | 18.4% |
| AUC | 97.5% | **97.8%** | 91.7% |

Calibration error (ECE) on the test voices: 0.025 before, 0.016 after isotonic calibration.

### 4.2 Confidence gate (test voices)

| "Correct" if P(error) ≤ | "Wrong" if ≥ | Sent to specialist | False rejection | Missed errors |
|---|---|---|---|---|
| 0.5 | 0.5 | 0.0% | 12.4% | 1.4% |
| 0.3 | 0.7 | 2.6% | 11.1% | 0.8% |
| 0.2 | 0.8 | 4.5% | 9.9% | 0.7% |
| **0.1** | **0.9** | **14.0%** | **4.9%** | **0.2%** |

Project targets: < 15% sent to specialist, < 10% false rejection. The 0.1 / 0.9 gate meets both **on this synthetic data**.

### 4.3 Each error type (test voices)

| Said instead of R | Caught | | Said instead of R | Caught |
|---|---|---|---|---|
| ع | 88.8% | | غ | 99.1% |
| ق | 95.7% | | م | 99.5% |
| خ | 96.7% | | ب د ذ ه ح ج ت ط ي ز | 100% |
| ن | 97.9% | | | |
| ل | 98.1% | | | |
| deleted | 98.5% | | | |
| و | 98.9% | | | |
| ك | 99.0% | | | |

### 4.4 Letters the model never saw

Five replacement letters (ز ق م ع خ) were removed from training and tested: **88.1%** of them caught (96.0% when trained on them). The model learned "this is not an R", not a list of specific letters.

### 4.5 In-between R sounds (never trained on)

| Blended towards | Amount | Mean P(error) | In the "unsure" zone (0.2–0.8) |
|---|---|---|---|
| ل | 0% (pure R) → 35% → 65% | 0.12 → 0.18 → 0.42 | 4% → 9% → 34% |
| غ | 0% → 35% → 65% | 0.12 → 0.18 → 0.37 | 4% → 12% → 34% |
| ي | 0% → 35% → 65% | 0.12 → 0.23 → 0.60 | 4% → 18% → 34% |
| و | 0% → 35% → 65% | 0.12 → 0.20 → 0.51 | 4% → 13% → 40% |

Confidence moves the right way: the further the R is blended, the higher P(error), and the more clips land in the zone that goes to the specialist.

### 4.6 By position and word

| R position | Balanced accuracy | False rejection |
|---|---|---|
| Start | 95.1% | 8.6% |
| Middle | 92.3% | 13.9% |
| End | 91.6% | 15.9% |

Weakest words: برتقال (88.8%), طائرة (89.5%), شجرة (90.0%), قرد (90.1%), رجل (90.2%).

### 4.7 Which features carry the result

Each feature family alone, trained and tested the same way:

| Family | Features | Balanced accuracy |
|---|---|---|
| `melgrid`: coarse time × frequency picture | 144 | 91.9% |
| `spec`: spectral shape | 36 | 87.4% |
| `mfcc` | 78 | 85.3% |
| `voice`: F0, voicing, HNR | 11 | 74.2% |
| `mod`: tongue trill (15–35 Hz ripple) | 10 | 72.8% |

The trill features are weaker than expected, likely because TTS voices mostly say a single-tap R rather than a rolled R.

---

## 5. Known problem: five voices

False rejection is **not** spread evenly. Almost all of it comes from five voices:

| Voice | Correct words marked wrong |
|---|---|
| ar-SY-Laith | 86% |
| ar-OM-Aysha | 85% |
| ar-SY-Amany (test voice) | 72% |
| ar-EG-Shakir | 58% |
| ar-OM-Abdullah | 55% |
| Other 24 Arabic voices | mostly 0–10% |

**Their R was checked by ear and is correct.** So the model is too strict: it learned one shape of R from most voices, and rejects a correct R said a different way. A hint (not proven): these voices have less 15–35 Hz trill ripple, so their R is closer to a single tap.

**Tried and did not help:** removing the voice's static colour from the features (subtracting the per-band mean over time). False rejection stayed 12.5–13.2% and the same voices stayed rejected ([`reports/exp_voice_robustness.log`](reports/exp_voice_robustness.log)).

**Why it matters:** real children's correct R will vary even more than these voices. This is the main thing to fix before testing on real children.

---

## 6. The model and its features

### Features: `scripts/extract_features.py`, 280 per clip

The model does not know where the R is, so every family is computed over the whole word **and** over 6 equal time segments.

| Family | Count | What |
|---|---|---|
| `melgrid` | 144 | Mean log-mel energy: 24 bands × 6 time segments |
| `mfcc` | 78 | 13 MFCC + delta + delta-delta, mean and std |
| `spec` | 36 | Spectral centroid, spread, rolloff, flatness, zero-crossing rate, energy (whole word and per segment) |
| `mod` | 10 | Modulation of the 300–3000 Hz envelope: share of 2–8 / 8–15 / 15–35 / 35–60 Hz, trill strength and position, peaks, dips |
| `voice` | 11 | F0 median and variation, voicing ratio, voiced regions, HNR (whole word and per segment) |
| `dur_s` | 1 | Duration |

Audio is RMS-normalised first, so loudness does not matter.

### Classifier: `scripts/train_baseline.py`

| | |
|---|---|
| Model | scikit-learn `HistGradientBoostingClassifier` (400 iterations, learning rate 0.08, 31 leaves, L2 = 1.0, balanced class weights) |
| Baseline | Standardisation + logistic regression |
| Calibration | Isotonic regression fitted on out-of-fold predictions |
| Saved as | `trained/r_baseline.joblib`: a dict with `model`, `calibrator`, `feature_names`, `positive_class` |

Load it in Python:

```python
import joblib, soundfile as sf
from scripts.extract_features import features

bundle = joblib.load("trained/r_baseline.joblib")
y, sr = sf.read("data/samples/w01_رمل_r.wav")          # 16 kHz mono
x = features(y)[None, :]
p_error = bundle["calibrator"].predict(bundle["model"].predict_proba(x)[:, 1])[0]
```

---

## 7. Rebuild everything

Run from this `model/` folder, in order. The scripts write into `data/raw/`, `data/blend/`, `data/child/`, `data/features/`.

| Step | Command | Time on a 4-core laptop |
|---|---|---|
| 1. Utterances | `python scripts/build_utterances.py` | seconds |
| 2. Adult TTS clips | `python scripts/generate_tts.py --engines edge --multilingual --rates "+0%,-20%" --errors-per-word 4` | ~40 min (internet) |
| 3. In-between R | `python scripts/make_blends.py` | ~25 min |
| 4. Child-like voices | `python scripts/childify.py --children-per-voice 4 --workers 16` | 1–3 h (disk-bound) |
| 5. Check | `python scripts/check_child.py --fix` then `childify.py` again, then `check_child.py --recheck` | ~10 min |
| 6. Split | `python scripts/make_splits.py` | seconds |
| 7. Features | `python scripts/extract_features.py --workers 4` | ~15 min |
| 8. Train + report | `python scripts/train_baseline.py` | ~10 min |

Listening pages for checking by ear: `python scripts/make_review.py` (writes an HTML page with the audio embedded).

Notes:
- The scripts ask Windows not to sleep while they run (they do not change any setting). A closed laptop lid still sleeps.
- TTS voices are an online service: generated audio can differ slightly between runs.
- `excluded_voices.txt`, `excluded_clips.txt` and the split come from checks by ear; rebuilding the audio does not redo those checks.

---

## 8. Files

```
model/
├── README.md                    this file
├── requirements.txt
├── trained/
│   └── r_baseline.joblib        the trained model + calibrator
├── scripts/
│   ├── predict.py               check the R in WAV files
│   ├── build_utterances.py      words -> correct + error utterances
│   ├── generate_tts.py          utterances -> adult TTS clips (edge / Azure / ElevenLabs)
│   ├── make_blends.py           in-between R sounds (DTW + WORLD vocoder)
│   ├── childify.py              adult -> child-like voices (Praat)
│   ├── check_child.py           automatic check of every clip
│   ├── make_splits.py           train / test / confidence split by voice
│   ├── extract_features.py      280 features per clip
│   ├── train_baseline.py        training, evaluation, report
│   ├── exp_robust.py            experiment: voice-colour removal (section 5)
│   ├── make_review.py           HTML listening pages
│   └── *.bat                    Windows runners for the long steps
├── data/
│   ├── words_r.csv
│   ├── utterances_r.csv
│   ├── splits.csv.gz
│   ├── excluded_voices.txt
│   ├── excluded_clips.txt
│   ├── metadata/                raw_tts.csv, blends.csv, child.csv.gz, child_check.csv
│   └── samples/                 12 example WAV files + samples.csv
└── reports/
    ├── baseline_report.md
    ├── baseline_metrics.json
    └── exp_voice_robustness.log
```

---

## 9. Next steps

1. **Fix the strict-R problem (section 5).** In progress: features from a pretrained multilingual speech model (mHuBERT-147, trained on 147 languages incl. Arabic), which should treat a tapped and a rolled R as the same sound.
2. **Verify the foreign voices** by ear and, if their R is correct, add them back to training.
3. **Test on real speech:** first the team's own recordings imitating errors, then children's recordings labelled by two specialists (needs ethics approval and parent consent).
4. **S sound (س)** with the same pipeline.
5. **Know the word:** in the real game the app knows which word it asked. Giving the model that information should make it simpler and more accurate.

---

*Part of [Nutq](../README.md) · MEDATHON 2026*
