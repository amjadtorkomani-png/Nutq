# Nutq · نُطق

**Speech-practice games for children, checked by AI and supervised by a speech therapist.**

ألعاب لتدريب النطق عند الأطفال، يفحصها الذكاء الاصطناعي ويشرف عليها أخصائي النطق.

> MEDATHON 2026 · Track 4: Out of Hospital Healthcare

---

## The problem

Speech therapy happens once a week. Practice has to happen every day.

- Children who mispronounce a sound need many short practice rounds between sessions.
- Specialists are hard to reach: sessions are weekly at best, and waiting lists are long outside the big cities.
- At home, nobody can tell right from wrong. Parents cannot judge the sound, so practice stops or repeats the error.

| | |
|---|---|
| **1 in 12** | children has a voice, speech, language or swallowing disorder (US data, NIDCD 2015) |
| **29 of 196** | major Saudi government hospitals offered speech therapy, in only 7 of 13 regions (national survey of SLP services) |
| **66%** | word error rate of a leading speech-to-text model on Arabic children's speech (Arabic Little STT, 2025) |

## Why speech-to-text is not enough

A child is asked to say **رَمْل** (sand) and says **غَمْل**.

- A speech-to-text app writes "رمل" and marks it correct. The error disappears.
- Nutq hears the R swapped for GH at the start of the word. The error is caught, named and tracked.

We already know which word the game asked for. So Nutq does not ask "what did the child say?". It asks **"was the target sound right?"**

## How it works

1. **The specialist sets the goal:** which letter, where in the word, and which words.
2. **The child plays at home:** short games that react to the child's voice.
3. **AI checks every word:** correct, close or wrong.
4. **Unsure? It goes to the specialist:** only unclear recordings need a human ear.

Every review by a specialist becomes new training data, so the checker improves with use.

### The checker ("the listener") in five steps

| Step | What it does | In the prototype |
|---|---|---|
| 1. Sound check | Too quiet, too short, too long? The child is asked to try again. | ✅ Recording warnings |
| 2. Right word? | Saying another word is caught, and is not judged as a speech error. | ✅ Word model |
| 3. Sound by sound | Names the sound the child said in place of the target. | ✅ Letter model |
| 4. Target letter | A focused model for the target sound. | ✅ Target-letter model |
| 5. Confidence gate | Sure: the result goes to the game. Unsure: "close" for the child, and the clip goes to the specialist. | ✅ صح / قريب / غلط |

## First release scope

- Children aged 5 to 9
- Every Arabic letter, one at a time, chosen by the specialist
- Saudi dialects
- Positioned as **supervised home training, not diagnosis**

## Project status

Built for MEDATHON 2026.

| Part | Status |
|---|---|
| Concept, pitch deck, business model | Done |
| **The listener** ([`model/`](model/)) | **Working prototype**: trained and tested |
| Child game app | Next phase |
| Specialist dashboard | Next phase |
| Data from real children | Planned, with ethics approval and parent consent |

### The listener today

The listener hears one word, knows which word the game asked for, and answers **صح**, **قريب** or **غلط** in about 0.2 seconds on a laptop.

- **How:** mHuBERT-147, a speech model pretrained on 147 languages, turns the recording into numbers that describe the sounds. Three small models then answer three questions: is it the right word, which sound is in the target letter's place, and is the target letter right.
- **Data:** no Arabic dataset of children's speech errors exists, so we generated one: 74,336 child-like clips with every error labelled. It covers the target letter at the start, middle and end of words, replaced by other letters or dropped, and in-between sounds.
- **Results** on 6 voices it never saw:

| What was said | Result |
|---|---|
| Errors | **99.0%** caught |
| Errors judged correct | **0.1%** |
| Correct words | **88.8%** accepted at once, 4.3% sent to the specialist |

Next milestone: validation with real children, labelled by speech specialists.

Full details, demo and results: **[model/README.md](model/README.md)**.

## Repository layout

```
model/              the listener (see model/README.md)
  trained/          the trained listener (nutq_r.joblib)
  scripts/          listener, training, and the synthetic-data pipeline
  data/             word list, utterances, split, metadata of all clips, audio samples
  reports/          test results
```

## Roadmap

| Phase | Goal |
|---|---|
| 0 · Prove the technology | Go/no-go: at least 75% accuracy on children's speech-sound errors |
| 1 · Collect data, build v1 | 100–150 children × 50 words, each clip labelled by two specialists |
| 2 · Field pilot | 30 children, 5 specialists, 8 weeks of home practice |
| 3 · Scale | Checking on the phone, sentences, more dialects |

Targets we will measure: ≥ 85% agreement with specialists, < 10% of correct words wrongly marked as errors, < 15% of recordings sent to the specialist, < 1.5 s from the end of the word to the result.

## Privacy and safety

- Parent consent before any recording.
- Children's voice data hosted inside Saudi Arabia, under PDPL.
- No real children's recordings are stored in this repository.
- Planned: application to the Ministry of Health regulatory sandbox before any commercial launch.

## Team

| Name | Role | Background |
|---|---|---|
| Hasan Akhras | Health lead | General Medicine, Umm Al-Qura University |
| Moayad Albahri | AI Engineer | Artificial Intelligence, University of Jeddah |
| Amjad Turkmani | AI Researcher | Artificial Intelligence, University of Jeddah |

