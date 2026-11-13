"""
Judge recordings from the command line.

Usage:
  python scripts/predict.py رمل recording.wav            (word as written, or its id: w01)
  python scripts/predict.py w01 a.wav b.wav --child      (--child: the speaker is already a child,
                                                          skip the child-voice conversion)
Output per file: صح / قريب / غلط, P(R wrong), the word that was heard and the sound in the R place.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import listener  # noqa: E402

AR = {"correct": "صح", "close": "قريب (للأخصائي)", "incorrect": "غلط"}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("word", help="the asked word, e.g. رمل or w01")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--child", action="store_true", help="speaker is a child: no voice conversion")
    a = ap.parse_args()
    words = {r["word_plain"]: r["word_id"] for r in csv.DictReader(open(listener.ROOT / "data" / "words_r.csv", encoding="utf-8"))}
    target = words.get(a.word, a.word)
    char = {r["produced"]: r["produced_char"] for r in csv.DictReader(open(listener.ROOT / "data" / "utterances_r.csv", encoding="utf-8"))}
    char["del"] = "لا شيء"
    L = listener.Listener()
    for f in a.files:
        r = L.judge(f, target_word=target, adult=not a.child)
        letter = char.get(r["letter"], r["letter"])
        why = {"word": " (كلمة ثانية)", "letter": f" (سمع «{letter}» مكان الراء)", "r": "", "": ""}[r["reason"]]
        print(f"{Path(f).name}: {AR[r['verdict']]}{why}  P(غلط)={r['p_error']:.2f}  الكلمة={r['heard_word']}  "
              f"الحرف={letter} ({r['p_letter']:.2f})")


if __name__ == "__main__":
    main()
