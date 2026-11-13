# Nutq Listener · سامع النطق: R

The Nutq "listener" hears one word said by a child, together with the word the child was asked to say, and answers:

| Answer | Meaning |
|---|---|
| **صح** (correct) | The child said the asked word and the **R (ر)** is right |
| **قريب** (close) | Not sure: the recording goes to the speech specialist |
| **غلط** (wrong) | A different word, or another sound in place of the R, or a wrong R |

> **Important:** the model is trained on **synthetic speech only** (text-to-speech voices turned into child-like voices). On synthetic test voices it is accurate. On real microphone recordings it is much weaker ([section 5.3](#53-real-recordings)). It is not ready for children. The next step is training and testing on real recordings labelled by speech specialists.

---

## Contents

1. [Quick start](#1-quick-start)
2. [How it works](#2-how-it-works)
3. [The data](#3-the-data)
4. [The model](#4-the-model)
5. [Results](#5-results)
6. [Rebuild everything](#6-rebuild-everything)
7. [Files](#7-files)
8. [Limits and next steps](#8-limits-and-next-steps)

---

## 1. Quick start

```bash
pip install -r requirements.txt
```

The first run downloads the speech model **mHuBERT-147** (about 360 MB) from Hugging Face.

### Command line

```bash
python scripts/predict.py رمل data/samples/w01_رمل_r.wav data/samples/w01_رمل_y.wav --child
```

- The first argument is the word the child was asked to say, written plainly or as its id (`w01`).
- `--child` means the speaker is already a child. Without it, the voice is first converted to a child-like voice, which is the setting for adult testers.

Output on the samples in this folder:

| File | What was said | Asked word | Answer |
|---|---|---|---|
| `w01_رمل_r.wav` | رَمْل | رمل | صح |
| `w01_رمل_y.wav` | يَمْل (R → ي) | رمل | غلط: heard ي in place of R |
| `w01_رمل_mix-y-35.wav` | R blended 35% towards ي | رمل | صح |
| `w01_رمل_mix-y-65.wav` | R blended 65% towards ي | رمل | غلط: heard ي in place of R |
| `w12_كرة_r.wav` | كُرَة | كرة | صح |
| `w12_كرة_l.wav` | كُلَة (R → ل) | كرة | غلط: heard ل in place of R |
| `w12_كرة_mix-l-35.wav` | R blended 35% towards ل | كرة | صح |
| `w12_كرة_mix-l-65.wav` | R blended 65% towards ل | كرة | صح |
| `w19_قمر_r.wav` | قَمَر | قمر | صح |
| `w19_قمر_gh.wav` | قَمَغ (R → غ) | قمر | غلط: heard غ in place of R |
| `w19_قمر_y.wav` | قَمَي (R → ي) | قمر | غلط: heard ي in place of R |
| `w19_قمر_mix-gh-65.wav` | R blended 65% towards غ | قمر | صح |
| `w01_رمل_r.wav` | رَمْل | **قمر** | غلط: a different word |

### Test app (your own voice)

```bash
python scripts/app.py
```

Then open **http://127.0.0.1:8765** in Chrome or Edge and allow the microphone.

How the app works:
- The page asks you to say a word, either correctly or with a given error (for example "say ل in place of R": لَمْل).
- It records you and shows the answer.
- It also shows whether the answer matches what you were asked to say, and keeps a score.
- Every recording is saved locally in `data/real/` with `log.csv`. This folder is in `.gitignore`: recordings of people's voices are never committed.

---

## 2. How it works

```
recording ─► 16 kHz, silence trimmed ─► child-like voice (adults only) ─► mHuBERT-147 ─► 4608 numbers
                                                                                            │
                     ┌──────────────────────────────────────────────────────────────────────┼─────────────────────────┐
                     ▼                                                                      ▼                         ▼
              word model                                                               letter model               R model
         which of the 25 words?                                                 which sound is in the R place?   P(R is wrong)
                     │                                                                      │                         │
                     └──────────────────────────────► decision ◄────────────────────────────┴─────────────────────────┘
```

### 2.1 Audio

| Step | What |
|---|---|
| Resample | Any sample rate → 16 kHz mono |
| Trim | Silence trimmed at start and end; 50 ms kept on each side |
| Child voice | Adult speakers only: Praat "Change gender" with formants ×1.2 and pitch 275 Hz, the middle of the ranges used to make the training voices |
| Loudness | RMS-normalised, so loudness does not matter |

### 2.2 Embedding: mHuBERT-147

[mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147) is a speech model pretrained on 147 languages, including Arabic. Its hidden layers encode **which sound is said** more than **how the voice sounds**. That matters here: a correct R said by different voices, tapped or rolled, should look the same.

Each recording becomes **4,608 numbers**:

| Part | Size |
|---|---|
| Mean over time of transformer layers 6, 9 and 12 | 3 × 768 |
| Mean of layer 9 in 3 equal time segments (start, middle, end of the word), so the model knows *where* a sound is | 3 × 768 |

### 2.3 Three models on the same numbers

All three are logistic regressions on standardised embeddings.

| Model | Question | Classes |
|---|---|---|
| **R model** | Is the R wrong? | correct / incorrect, probability calibrated |
| **Word model** | Which word was said? | 25 words; an error version counts as its word (لَمْل → رمل) |
| **Letter model** | Which sound is in the R place? | ر + 19 letters (غ ل ي و ب د ن ت ع ج م ه خ ق ك ز ذ ط ح) + "no R" |

### 2.4 Decision

Checked in this order:

| # | Rule | Answer |
|---|---|---|
| 1 | The word model hears another word, and gives the asked word less than 25% | **غلط** |
| 2 | The letter model hears another sound than R, with at least 50% | **غلط** |
| 3 | P(R wrong) < 0.1 | **صح** |
| 4 | P(R wrong) > 0.9 | **غلط** |
| 5 | Otherwise | **قريب**: goes to the specialist |

### Why not speech-to-text?

Take a child who says **غَمْل** instead of **رَمْل**:
- Speech-to-text writes "رمل" and marks it correct.
- The listener judges the **sound**, so it answers غلط.

---

## 3. The data

No recordings of real children are used or stored. All training data is generated.

### 3.1 How it is made

```
25 words  ──►  497 utterances  ──►  TTS (edge-tts)  ──►  in-between R  ──►  child-like voice  ──►  automatic check
              (correct + errors)    44 voices × 2 speeds   (WORLD vocoder)    (Praat, 4 per voice)
```

**Words.** [`data/words_r.csv`](data/words_r.csv) has 25 everyday words a child knows, with R at the start (9), middle (9) and end (7):

رمل، رجل، رأس، رمان، ريشة، رقم، رسم، روبوت، ربيع، أرنب، برتقال، كرة، شجرة، سيارة، طائرة، بقرة، فراشة، قرد، قمر، بحر، نمر، تمر، حجر، نهر، صقر

Rules for the list:
- One R per word.
- No غ in the word.
- No two words that differ only in the R.

**Utterances.** [`data/utterances_r.csv`](data/utterances_r.csv), made by `build_utterances.py`. For every word it holds:
- the correct form,
- the R replaced by each of 19 letters,
- the R deleted.

A variant is dropped if it can't be pronounced or if it doubles a neighbouring letter. Result: **497 utterances (25 correct, 472 errors)**, all with full diacritics. For the TTS only, a final ة is written as ه so that the voices pronounce it.

**Adult voices.** `generate_tts.py`, with edge-tts:
- 32 Arabic voices from 16 countries, plus 12 multilingual voices, each at normal speed and 20% slower.
- Per word and voice: the correct form plus 4 sampled errors.
- **11,085 clips**.

**In-between R.** `make_blends.py`. Children often say a sound *between* R and another letter.
- The correct clip and an error clip (ل غ ي و) of the same voice are aligned with DTW.
- The region where they differ (the R) is found.
- Inside that region, the WORLD vocoder spectral envelope is mixed at 35% and 65%.
- The originals are also passed through the vocoder, so vocoder artefacts appear in every class.
- **7,499 clips**.

**Child-like voices.** `childify.py`, with Praat "Change gender". Each source voice becomes 4 "virtual children", each with fixed settings:

| Setting | Range | Why |
|---|---|---|
| Formant shift | ×1.12 to ×1.28 | Shorter vocal tract |
| Pitch (F0) | 230–320 Hz | Children's voice range |
| Pitch range | ×0.8 to ×1.4 | Flat or lively intonation |
| Speed | ×1.00 to ×1.25 | Children often speak slower |

Result: **74,336 child-like clips**.

**Automatic check.** `check_child.py` flags a clip that is broken, too short, has no voiced part, or has F0 outside 180–400 Hz. 170 clips are excluded ([`data/excluded_clips.txt`](data/excluded_clips.txt)).

**Checked by ear.** Three voices (ar-AE-Fatima, ar-LB-Layla, ar-SA-Zariyah) say "ارمل" for رمل and "عجل" for رجل. They are excluded ([`data/excluded_voices.txt`](data/excluded_voices.txt)).

### 3.2 The split

[`data/splits.csv.gz`](data/splits.csv.gz), made by `make_splits.py`. A voice is never on both sides: all its speeds and virtual children stay together.

| Set | Clips | Content | Used for |
|---|---|---|---|
| `train` | 32,834 | 23 Arabic voices, 5 folds by voice | Training and calibration |
| `test_voices` | 8,503 | 6 Arabic voices (3 male, 3 female) never seen in training | Test |
| `confidence` | 14,931 | In-between R sounds | Not trained on |
| `held_foreign` | 11,975 | Multilingual voices | Held out: their R is not verified |
| `excluded` | 6,093 | Excluded voices and clips | Not used |

### 3.3 What is in this folder

| File | Rows | Content |
|---|---|---|
| `data/words_r.csv` | 25 | Word list |
| `data/utterances_r.csv` | 497 | Every utterance with label, replaced letter and TTS text |
| `data/splits.csv.gz` | 74,336 | Set and fold of every clip |
| `data/metadata/raw_tts.csv` | 11,085 | Adult TTS clips |
| `data/metadata/blends.csv` | 7,499 | In-between clips: blend amount and R region |
| `data/metadata/child.csv.gz` | 74,336 | Child-like clips: source clip and voice settings |
| `data/metadata/child_check.csv` | 170 | Clips flagged by the automatic check |
| `data/samples/` | 12 | Example WAV files (see `samples.csv`) |

**Not in the repository:** the audio (1.5 GB) and the embeddings (520 MB). The scripts rebuild both ([section 6](#6-rebuild-everything)).

---

## 4. The model

All three classifiers are trained by `scripts/train_model.py` and saved in **one file**: [`trained/nutq_r.joblib`](trained/nutq_r.joblib) (1.2 MB). The speech model itself is not in it; it is loaded from Hugging Face.

| Part | Training |
|---|---|
| `r_model` | StandardScaler + LogisticRegression (C = 0.05, balanced classes), all 32,834 train clips |
| `calibrator` | Isotonic regression on 5-fold out-of-fold predictions (folds = voices), so P(R wrong) is a real probability |
| `word_model` | StandardScaler + LogisticRegression (C = 0.02), up to 28 clips per utterance (13,050 clips), no in-between clips |
| `letter_model` | Same as `word_model`, with the replaced letter as the label |
| `thresholds` | `lo` 0.1, `hi` 0.9, `word_min` 0.25, `letter_min` 0.5 |

Use it in Python:

```python
import sys; sys.path.insert(0, "scripts")
import listener

L = listener.Listener()                                    # loads mHuBERT-147 + trained/nutq_r.joblib
r = L.judge("data/samples/w01_رمل_y.wav", target_word="w01", adult=False)
print(r["verdict"], r["reason"], round(r["p_error"], 2), r["heard_word"], r["letter"])
# incorrect letter 1.0 w01 y
```

`reason` says why the answer is غلط:
- `word`: a different word
- `letter`: another sound in the R place
- `r`: the R model

---

## 5. Results

All numbers are in [`reports/model_metrics.json`](reports/model_metrics.json).

### 5.1 Test: 6 synthetic voices the model never saw (8,503 clips)

**The whole listener**, each clip judged against its own word:

| Clips | صح | قريب | غلط |
|---|---|---|---|
| Correct words (R right) | **88.8%** | 4.3% | 6.9% |
| Words with an error | 0.1% | 1.0% | **99.0%** |
| Correct words, but a different word was asked | 0.0% | 0.1% | **99.9%** |

When a correct word is wrongly marked غلط, most of these come from the letter rule (97 clips). The word rule causes 30 and the R model 22.

### 5.2 Each model alone (same test voices)

| Model | Measure | Result |
|---|---|---|
| R model | Balanced accuracy (at P = 0.5) | 96.8% |
| | Correct words marked wrong | 5.5% |
| | Errors missed | 0.9% |
| | AUC | 99.0% |
| Word model | Right word | 97.7% |
| Letter model | Right sound in the R place (all clips) | 86.0% |
| | Right sound, error clips only | 84.0% |

The letter model mostly confuses sounds that are close:

| Said | Taken for |
|---|---|
| ذ | ز |
| م | ن |
| ط | ت |
| ق | no R |

### 5.3 Real recordings

These are 54 recordings made with the test app by one adult team member on a laptop microphone:
- 45 asked to be said correctly,
- 9 asked with a given error.

| Asked to say | صح | قريب | غلط |
|---|---|---|---|
| Correctly (45) | 28.9% | 8.9% | **62.2%** |
| With an error (9) | 22.2% | 0.0% | **77.8%** |

On real recordings, most correct words are marked wrong. The models learned from synthetic voices only, and a real voice through a real microphone is different from them. **Closing this gap is the main work before any use with children.**

---

## 6. Rebuild everything

Run from this `model/` folder, in order. The scripts write into `data/raw/`, `data/blend/`, `data/child/` and `data/features/`.

| Step | Command | Time on a 4-core laptop |
|---|---|---|
| 1. Utterances | `python scripts/build_utterances.py` | seconds |
| 2. Adult TTS clips | `python scripts/generate_tts.py --engines edge --multilingual --rates "+0%,-20%" --errors-per-word 4` | ~40 min (internet) |
| 3. In-between R | `python scripts/make_blends.py` | ~25 min |
| 4. Child-like voices | `python scripts/childify.py --children-per-voice 4 --workers 16` | 1–3 h (disk-bound) |
| 5. Check | `python scripts/check_child.py --fix`, then `childify.py` again, then `check_child.py --recheck` | ~10 min |
| 6. Split | `python scripts/make_splits.py` | seconds |
| 7. Embeddings | `python scripts/extract_embeddings.py`, then `python scripts/extract_embeddings.py --merge` | ~3–4 h on CPU (resumable) |
| 8. Train + measure | `python scripts/train_model.py` | ~10 min |

Notes:
- `make_review.py` writes HTML listening pages (audio embedded) for checking clips by ear.
- The scripts ask Windows not to sleep while they run. A closed laptop lid still sleeps.
- TTS voices are an online service, so the audio can differ slightly between runs.
- `excluded_voices.txt` and `excluded_clips.txt` come from checks by ear. Rebuilding the audio does not redo them.

---

## 7. Files

```
model/
├── README.md                    this file
├── requirements.txt
├── run_app.bat                  Windows: start the test app and open it
├── trained/
│   └── nutq_r.joblib            r_model + calibrator + word_model + letter_model + thresholds
├── scripts/
│   ├── listener.py              audio → embedding → 3 models → صح / قريب / غلط
│   ├── predict.py               judge WAV files from the command line
│   ├── app.py                   local test app (server)
│   ├── train_model.py           training + measurement → trained model, reports/model_metrics.json
│   ├── extract_embeddings.py    mHuBERT-147 embeddings of all clips
│   ├── build_utterances.py      words → correct + error utterances
│   ├── generate_tts.py          utterances → adult TTS clips
│   ├── make_blends.py           in-between R (DTW + WORLD vocoder)
│   ├── childify.py              adult → child-like voices (Praat)
│   ├── check_child.py           automatic check of every clip
│   ├── make_splits.py           train / test / confidence split by voice
│   ├── make_review.py           HTML listening pages
│   └── *.bat                    Windows runners for the long data steps
├── app/
│   └── index.html               the test app page
├── data/
│   ├── words_r.csv, utterances_r.csv, splits.csv.gz
│   ├── excluded_voices.txt, excluded_clips.txt
│   ├── metadata/                raw_tts.csv, blends.csv, child.csv.gz, child_check.csv
│   └── samples/                 12 example WAV files + samples.csv
└── reports/
    └── model_metrics.json
```

---

## 8. Limits and next steps

| Limit | Next step |
|---|---|
| Trained on synthetic voices only; weak on real recordings ([5.3](#53-real-recordings)) | Train with real recordings: first the team's own (collected with the test app), then children's recordings labelled by two specialists (needs ethics approval and parent consent) |
| The letter rule rejects some correct words | Tune `letter_min` on real recordings |
| One target sound | S (س) with the same pipeline |
| Runs on a laptop CPU (~0.2 s per word) | Smaller model for checking on the phone |

---

*Part of [Nutq](../README.md) · MEDATHON 2026*
