# Nutq · نُطق

**Speech-practice games for children, checked by AI and supervised by a speech therapist.**

ألعاب لتدريب النطق عند الأطفال، يفحصها الذكاء الاصطناعي ويشرف عليها أخصائي النطق.

> MEDATHON 2026 · Track 4: Out of Hospital Healthcare

---

## The problem

Speech therapy happens once a week. Practice has to happen every day.

- Children who say the R or S sound the wrong way need many short practice rounds between sessions.
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

We already know which word the game asked for. So Nutq does not ask "what did the child say?". It asks **"was the R sound right?"**

## How it works

1. **The specialist sets the goal:** which letter, where in the word, and which words.
2. **The child plays at home:** the game reacts to the voice (the car speeds up as the child rolls the R).
3. **AI checks every word:** correct, close or wrong.
4. **Unsure? It goes to the specialist:** only unclear recordings need a human ear.

Every review by a specialist becomes new training data, so the checker improves with use.

### The checker ("the listener") in five steps

| Step | What it does |
|---|---|
| 1. Sound check | Too quiet or noisy? The game asks the child to move closer. A bad recording is never judged. |
| 2. Right word? | Laughing or saying another word does not count as a speech error. |
| 3. Sound by sound | Compares the sounds spoken with the sounds expected. |
| 4. Target letter | A focused model for the target sound. |
| 5. Confidence gate | Sure: the result goes to the game. Unsure: "close" for the child, and the clip goes to the specialist. |

## First release scope

- Children aged 5 to 9
- The R and S sounds
- Saudi dialects
- Positioned as **supervised home training, not diagnosis**

## Project status

This is an early-stage prototype built for MEDATHON 2026.

| Part | Status |
|---|---|
| Concept, pitch deck, business model | Done |
| Listener for the R sound (`nutq-listener/`) | Proof of concept in progress |
| Child game app | Not started |
| Specialist dashboard | Not started |
| Data from real children | Not collected yet (needs ethics approval and parent consent) |

**About the current listener:** it is trained and tested on **synthetic speech only** (text-to-speech voices converted to child-like voices, with the R replaced by other letters). Results on synthetic data say nothing yet about real children. The next milestone is a test on real recordings labelled by speech specialists.

## Repository layout

```
nutq-listener/      R-sound checker: data generation, features, training, reports
  scripts/          pipeline scripts (TTS generation, child-voice conversion, features, training)
  data/             word lists and generated data (audio is not committed)
  reports/          evaluation reports
```

## Roadmap

| Phase | Goal |
|---|---|
| 0 · Prove the technology | Go/no-go: at least 75% accuracy on children's R and S errors |
| 1 · Collect data, build v1 | 100–150 children × 50 words, each clip labelled by two specialists |
| 2 · Field pilot | 30 children, 5 specialists, 8 weeks of home practice |
| 3 · Scale | Checking on the phone, more letters, sentences |

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

