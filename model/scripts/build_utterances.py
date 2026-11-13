"""
Build data/utterances_r.csv from data/words_r.csv.

For every word: one correct utterance (ر) plus every error variant:
  - substitution: ر replaced by another letter (diacritics kept)
  - deletion:     ر removed (word-initial deletion becomes a hamza, e.g. رَمْل -> أَمْل)

label    = correct | incorrect
produced = short ASCII code of what is said instead of ر (r = correct)
tier     = common | other  (common = errors children often make; used for sampling weights)

A variant is dropped if, without diacritics, it equals a word in the list
(that would break the closed-vocabulary assumption), or if the new letter
doubles a neighbouring letter (ambiguous for TTS).

Usage: python scripts/build_utterances.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARAKAT = {chr(c) for c in range(0x064B, 0x0653)}  # tanween .. sukun, shadda

SUBS = [  # code, letter, tier
    ("gh", "غ", "common"), ("l", "ل", "common"), ("y", "ي", "common"), ("w", "و", "common"),
    ("b", "ب", "other"), ("d", "د", "other"), ("n", "ن", "other"), ("t", "ت", "other"),
    ("ain", "ع", "other"), ("j", "ج", "other"), ("m", "م", "other"), ("h", "ه", "other"),
    ("kh", "خ", "other"), ("q", "ق", "other"), ("k", "ك", "other"), ("z", "ز", "other"),
    ("dh", "ذ", "other"), ("tt", "ط", "other"), ("hh", "ح", "other"),
]


def strip(s):
    return "".join(c for c in s if c not in HARAKAT)


def r_span(text):
    """index of ر and the end of its diacritics"""
    i = text.index("ر")
    j = i + 1
    while j < len(text) and text[j] in HARAKAT:
        j += 1
    return i, j


def neighbours(text, i, j):
    base = [c for c in text[:i] if c not in HARAKAT]
    after = [c for c in text[j:] if c not in HARAKAT]
    return (base[-1] if base else None), (after[0] if after else None)


def main():
    with open(ROOT / "data" / "words_r.csv", encoding="utf-8") as f:
        words = list(csv.DictReader(f))
    vocab = {w["word_plain"] for w in words}
    out, dropped = [], []

    for w in words:
        t = w["correct_text"]
        i, j = r_span(t)
        prev, nxt = neighbours(t, i, j)
        base = dict(word_id=w["word_id"], position=w["position"], word_plain=w["word_plain"])
        out.append(dict(base, utt_id=f"{w['word_id']}_r", label="correct", produced="r",
                        produced_char="ر", tier="correct", text=t))

        for code, ch, tier in SUBS:
            nt = t[:i] + ch + t[i + 1:]
            if strip(nt) in vocab or ch in (prev, nxt):
                dropped.append(f"{w['word_plain']}->{strip(nt)}")
                continue
            out.append(dict(base, utt_id=f"{w['word_id']}_{code}", label="incorrect", produced=code,
                            produced_char=ch, tier=tier, text=nt))

        # deletion. Skipped for a voweled medial ر (كُرَة -> كُة is not pronounceable).
        r_vowel = any(c in HARAKAT and c not in ("ْ", "ّ") for c in t[i + 1:j])
        medial_voweled = i > 0 and nxt is not None and r_vowel
        if i == 0:
            seat = "إ" if "ِ" in t[i + 1:j] else "أ"  # kasra -> hamza below
            nt = (seat + t[i + 1:]).replace("أَأْ", "آ")      # رَأْس -> آس
        else:
            nt = t[:i] + t[j:]
        if not medial_voweled and strip(nt) not in vocab:
            out.append(dict(base, utt_id=f"{w['word_id']}_del", label="incorrect", produced="del",
                            produced_char="-", tier="common", text=nt))

    # edge TTS often drops the final vowel of a word ending in ة (فراشة -> "فراش", كرة -> "كُر").
    # A final ه is read as the pausal "a" sound (checked by ear, 2026-11-12), so every word ending
    # in ة is written with ه for the TTS only. word_plain keeps the real spelling.
    for r in out:
        if r["word_plain"].endswith("ة") and r["text"].endswith("ة"):
            r["text"] = r["text"][:-1] + "ه"

    fields = ["utt_id", "word_id", "position", "word_plain", "label", "produced",
              "produced_char", "tier", "text"]
    with open(ROOT / "data" / "utterances_r.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(out)

    n_inc = sum(r["label"] == "incorrect" for r in out)
    print(f"{len(out)} utterances: {len(out) - n_inc} correct, {n_inc} incorrect; dropped {len(dropped)}")
    if dropped:
        print("  dropped:", ", ".join(dropped))


if __name__ == "__main__":
    main()
