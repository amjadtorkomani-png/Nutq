# The Nutq Listener · سامع النطق

**The AI inside Nutq that hears a child say a word and decides whether the target letter was said right.**

سامع النطق هو الذكاء الاصطناعي داخل نُطق. يسمع الطفل وهو يقول الكلمة اللي طلبتها اللعبة، ويحكم: **صح**، أو **قريب** (يتحوّل للأخصائي)، أو **غلط**.

> Part of [Nutq · نُطق](../README.md), MEDATHON 2026 · Track 4: Out of Hospital Healthcare

---

## At a glance

| | |
|---|---|
| **Input** | A recording of one word + the word the game asked for |
| **Output** | **صح** (correct) · **قريب** (close: sent to the specialist) · **غلط** (wrong) |
| **Letters** | Built to cover every Arabic letter. The examples and results below use ر (R), in 25 everyday words, at the start, middle and end of the word |
| **Errors it catches** | The target letter replaced by another letter, the letter dropped, or a different word |
| **Core technology** | mHuBERT-147, a speech model pretrained on 147 languages, + three small classifiers |
| **Training data** | Synthetic child-like speech: 74,336 clips generated from Arabic text-to-speech voices of 16 countries; 32,834 used for training |
| **Speed** | About 0.2 s per word on a laptop CPU, no GPU |
| **Result** | On 6 voices it never saw: **99.0%** of errors caught, **88.8%** of correct words accepted at once |

---

## Where it fits in Nutq

```
   Specialist                      Child at home                         The listener                       Specialist
 sets the goal    ──►    plays the game and says the word    ──►    صح / قريب / غلط    ──►    reviews only the "قريب" clips
 (letter, words)          (the game knows the asked word)          in about 0.2 seconds          and tracks progress
```

The game always knows which word it asked for. So the listener does not ask *"what did the child say?"*. It asks **"is this the right word, and is the target letter right?"**

That is why plain speech-to-text cannot do this job. A child asked for **رَمْل** says **غَمْل**:
- **Speech-to-text** hears "رمل" and calls it correct.
- **The listener** hears غ where the R should be and answers **غلط**.

---

## Try it

```bash
pip install -r requirements.txt
python scripts/predict.py قمر data/samples/w19_قمر_r.wav data/samples/w19_قمر_gh.wav --child
```

The first run downloads mHuBERT-147 (about 360 MB). The first argument is the asked word. `--child` means the speaker is a child. Without it, the voice is first converted to a child-like one, which is the setting for adult testers.

| Sample | What was said | Asked word | Answer |
|---|---|---|---|
| `w19_قمر_r.wav` | قَمَر | قمر | **صح** |
| `w19_قمر_gh.wav` | قَمَغ (R → غ) | قمر | **غلط**: غ in place of R |
| `w19_قمر_y.wav` | قَمَي (R → ي) | قمر | **غلط**: ي in place of R |
| `w12_كرة_r.wav` | كُرَة | كرة | **صح** |
| `w12_كرة_l.wav` | كُلَة (R → ل) | كرة | **غلط**: ل in place of R |
| `w01_رمل_r.wav` | رَمْل | رمل | **صح** |
| `w01_رمل_y.wav` | يَمْل (R → ي) | رمل | **غلط**: ي in place of R |
| `w01_رمل_mix-y-65.wav` | R drifting 65% towards ي | رمل | **غلط**: ي in place of R |
| `w01_رمل_r.wav` | رَمْل | **قمر** | **غلط**: a different word |

---

## How it works

```
 recording
    │   16 kHz · silence trimmed · loudness normalised · child-like voice (adult testers only)
    ▼
 mHuBERT-147  ──►  4,608 numbers that describe which sounds were said
    │
    ├──►  Word model      which of the 25 words was said?
    ├──►  Letter model    which sound is in the R's place?   (ر, one of 19 letters, or nothing)
    └──►  R model         how likely is the R wrong?        (calibrated probability)
                │
                ▼
            Decision
```

### The decision

| Check | Rule | Answer |
|---|---|---|
| 1. Right word? | Another word is heard, and the asked word gets less than 25% | **غلط** |
| 2. Right sound in the R's place? | Another sound is heard there, with at least 50% | **غلط** |
| 3. How sure is the R model? | P(R wrong) below 0.1 | **صح** |
| | P(R wrong) above 0.9 | **غلط** |
| | Anything between | **قريب**: the clip goes to the specialist |

The **قريب** zone is a deliberate safety valve. When the listener is unsure, a human decides. This protects the child from being told "wrong" on a correct word, and it gives the specialist the clips that matter most.

### Why these choices

| Choice | Why |
|---|---|
| **Synthetic training data** | There is no public dataset of Arabic children's speech errors. We generate one, with every error labelled by construction. |
| **Child-like voices** | Adult text-to-speech voices are converted with Praat. Formants are raised (shorter vocal tract) and pitch is set to 230–320 Hz, so the model learns child acoustics. |
| **In-between sounds** | Children often say a sound *between* R and another letter. We create these by blending the R region of two recordings, so the model sees mild errors, not only clear ones. |
| **mHuBERT-147** | Pretrained on 147 languages including Arabic. Its inner layers capture *which sound* is said rather than *whose voice* it is, so a correct R from a new voice still looks correct. |
| **The asked word as input** | The game knows the word. Using it makes the job simpler and catches a child who says another word. |
| **Three small, separate models** | Each answers one clear question. Each can be checked on its own, and each is fast enough for a laptop. |
| **A confidence gate** | Unsure cases go to the specialist instead of being guessed. |

---

## Results

Tested on **6 voices (3 male, 3 female) that no model saw in training**: 8,503 clips. A voice is never on both sides of the split. All its speeds and child versions stay together.

### The whole listener

| What was said | صح | قريب | غلط |
|---|---|---|---|
| The asked word, R correct | **88.8%** | 4.3% | 6.9% |
| The asked word, R wrong | 0.1% | 1.0% | **99.0%** |
| A different word | 0.0% | 0.1% | **99.9%** |

Only **0.1%** of errors are let through as correct.

### Mild errors are handled with care

The R region is blended towards another letter. These blends were **never used in training**.

| How far the R drifts | Mean P(R wrong) | صح | قريب | غلط |
|---|---|---|---|---|
| 0%: clean R | 0.07 | **85.9%** | 4.6% | 9.6% |
| 35% towards another letter | 0.20 | 59.3% | 14.1% | 26.6% |
| 65% towards another letter | 0.69 | 14.9% | 12.5% | **72.5%** |
| 100%: clear error | 0.97 | 0.1% | 1.4% | **98.5%** |

The further the R drifts, the more the listener leans to **غلط** and the more it sends to the specialist. It does not jump between extremes. This is the behaviour a specialist expects.

### Each model on its own

| Model | Measure | Result |
|---|---|---|
| R model | AUC | 99.0% |
| | Balanced accuracy | 96.8% |
| Word model | Right word out of 25 | 97.7% |
| Letter model | Right sound in the R's place, out of 21 | 86.0% |

The next milestone is validation with real children, labelled by speech specialists ([next steps](#next-steps)).

---

## The training data

Every clip is generated. No recordings of real children are used or stored.

```
25 words ─► 497 utterances ─► 44 TTS voices × 2 speeds ─► in-between R ─► 4 child voices each ─► automatic check
           (1 correct + up to    (edge-tts)                  (WORLD vocoder)    (Praat)           (74,336 clips)
            20 errors per word)
```

| Step | Result |
|---|---|
| **25 words** a child knows: رمل، رجل، رأس، رمان، ريشة، رقم، رسم، روبوت، ربيع، أرنب، برتقال، كرة، شجرة، سيارة، طائرة، بقرة، فراشة، قرد، قمر، بحر، نمر، تمر، حجر، نهر، صقر | R at the start (9), middle (9) and end (7) |
| **497 utterances**: each word correct, with R replaced by 19 letters, and with R dropped, all fully voweled | 25 correct, 472 errors |
| **Adult voices**: 32 Arabic voices from 16 countries + 12 multilingual voices, normal speed and 20% slower | 11,085 clips |
| **In-between R**: correct and error recordings aligned (DTW); the R region blended at 35% and 65% with the WORLD vocoder | 7,499 clips |
| **Child-like voices**: each voice becomes 4 "virtual children" (formants ×1.12–1.28, pitch 230–320 Hz, speed ×1.00–1.25) | 74,336 clips |
| **Quality checks**: automatic (broken, too short, unvoiced, pitch out of range) and by ear | 170 clips and 3 voices removed |

<details>
<summary><b>Split, files and details</b></summary>

| Set | Clips | Use |
|---|---|---|
| `train` | 32,834 | Training: 23 Arabic voices, 5 folds by voice |
| `test_voices` | 8,503 | Test: 6 Arabic voices never seen in training |
| `confidence` | 14,931 | In-between R of all voices; never trained on |
| `held_foreign` | 11,975 | Multilingual voices, kept for later validation |
| `excluded` | 6,093 | Removed by the quality checks |

| File | Content |
|---|---|
| [`data/words_r.csv`](data/words_r.csv) | The 25 words |
| [`data/utterances_r.csv`](data/utterances_r.csv) | All 497 utterances, with label and replaced letter |
| [`data/splits.csv.gz`](data/splits.csv.gz) | Set and fold of every clip |
| [`data/metadata/`](data/metadata/) | Every clip: voice, speed, text, blend amount, child-voice settings, check results |
| [`data/samples/`](data/samples/) | 12 example clips |
| [`data/excluded_voices.txt`](data/excluded_voices.txt), [`data/excluded_clips.txt`](data/excluded_clips.txt) | Removed by the checks |

The audio (1.5 GB) and the embeddings (520 MB) are rebuilt by the scripts, so they are not stored in the repository.

</details>

---

## The model file

[`trained/nutq_r.joblib`](trained/nutq_r.joblib) (1.2 MB) holds everything except mHuBERT-147 itself, which is loaded from Hugging Face.

| Part | What |
|---|---|
| `r_model` | Standardisation + logistic regression (C = 0.05, balanced classes), trained on all 32,834 training clips |
| `calibrator` | Isotonic regression on out-of-fold predictions, so P(R wrong) is a true probability |
| `word_model` | Standardisation + logistic regression (C = 0.02), 25 words |
| `letter_model` | Standardisation + logistic regression (C = 0.02), 21 classes |
| `thresholds` | 0.1 and 0.9 (confidence gate), 0.25 (word), 0.5 (letter) |

**Embedding layout** (4,608 numbers per clip):
- mean over time of mHuBERT layers 6, 9 and 12 (3 × 768),
- layer 9 averaged over the start, middle and end of the word (3 × 768), so the model knows *where* each sound is.

```python
import sys; sys.path.insert(0, "scripts")
import listener

L = listener.Listener()
r = L.judge("data/samples/w01_رمل_y.wav", target_word="w01", adult=False)
print(r["verdict"], r["reason"], round(r["p_error"], 2), r["letter"])
# incorrect letter 1.0 y
```

`reason` tells the game *why* the answer is غلط:
- `word`: a different word
- `letter`: another sound in the R's place
- `r`: the R model

The game can use this to give the child the right hint.

<details>
<summary><b>Rebuild everything from scratch</b></summary>

Run from this `model/` folder, in order:

| Step | Command | Time (4-core laptop) |
|---|---|---|
| 1. Utterances | `python scripts/build_utterances.py` | seconds |
| 2. Adult TTS | `python scripts/generate_tts.py --engines edge --multilingual --rates "+0%,-20%" --errors-per-word 4` | ~40 min |
| 3. In-between R | `python scripts/make_blends.py` | ~25 min |
| 4. Child voices | `python scripts/childify.py --children-per-voice 4 --workers 16` | 1–3 h |
| 5. Checks | `python scripts/check_child.py --fix`, `childify.py` again, `check_child.py --recheck` | ~10 min |
| 6. Split | `python scripts/make_splits.py` | seconds |
| 7. Embeddings | `python scripts/extract_embeddings.py`, then `--merge` | 3–4 h on CPU, resumable |
| 8. Train + test | `python scripts/train_model.py` | ~10 min |

`make_review.py` writes HTML listening pages for checking clips by ear.

</details>

<details>
<summary><b>Files</b></summary>

```
model/
├── README.md
├── requirements.txt
├── trained/nutq_r.joblib        the trained listener
├── reports/model_metrics.json   all test numbers
├── scripts/
│   ├── listener.py              audio → mHuBERT-147 → 3 models → صح / قريب / غلط
│   ├── predict.py               judge WAV files from the command line
│   ├── train_model.py           training and testing
│   ├── extract_embeddings.py    mHuBERT-147 embeddings for all clips
│   ├── build_utterances.py      words → correct and error utterances
│   ├── generate_tts.py          utterances → adult voices
│   ├── make_blends.py           in-between R
│   ├── childify.py              adult → child-like voices
│   ├── check_child.py           automatic quality check
│   ├── make_splits.py           split by voice
│   ├── make_review.py           listening pages
│   └── *.bat                    Windows runners for the long steps
└── data/                        words, utterances, split, metadata, samples
```

</details>

---

## Next steps

| Step | Goal |
|---|---|
| **1. Real children** | Recordings of children, each labelled by two speech specialists (with ethics approval and parent consent). Fine-tune and measure against the targets: ≥ 85% agreement with specialists, < 10% of correct words marked wrong, < 15% sent to the specialist. |
| **2. Learn from every review** | Each clip a specialist reviews in the dashboard becomes new training data. |
| **3. Every Arabic letter** | The same pipeline for each letter: a new word list and generated errors, the same models. |
| **4. On the phone** | A smaller model, so checking runs on the device and the voice never leaves it. |

---

*Part of [Nutq · نُطق](../README.md) · MEDATHON 2026*
