"""
Fix the data split for the R task (سامع النطق). Writes data/splits.csv.

Every child clip gets one `set`:
  train        Arabic voices, correct + incorrect (incl. vocoded copies). 5 folds, split by voice.
  test_voices  same kind of clips from 6 Arabic voices the model never sees in training.
  confidence   "in-between" R clips (distortion). NOT trained on: used to test that the model's
               confidence drops as the R is blended towards another sound.
  held_foreign clips of the foreign (Multilingual) voices. Their correct clips are unverified,
               and training on their errors only would teach "foreign accent = error".
  excluded     excluded voices (data/excluded_voices.txt) and clips (data/excluded_clips.txt).

A voice (voice_group) is never on both sides: all its speeds and virtual children stay together.

Usage: python scripts/make_splits.py
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 7
N_TEST_PER_GENDER = 3
N_FOLDS = 5

FEMALE = {"Laila", "Amina", "Salma", "Rana", "Sana", "Noura", "Iman", "Mouna", "Aysha", "Amal",
          "Amany", "Reem", "Maryam", "Fatima", "Layla", "Zariyah"}


def read_list(name):
    p = ROOT / "data" / name
    if not p.exists():
        return set()
    return {l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}


def gender(voice_group):
    name = voice_group.split("-")[2].replace("Neural", "")
    return "f" if name in FEMALE else "m"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    bad_voices, bad_clips = read_list("excluded_voices.txt"), read_list("excluded_clips.txt")
    with open(ROOT / "data" / "child" / "metadata.csv", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))

    arabic = sorted({m["voice_group"] for m in meta if m["locale"].startswith("ar-")} - bad_voices)
    rng = random.Random(SEED)
    by_gender = defaultdict(list)
    for v in arabic:
        by_gender[gender(v)].append(v)
    test_voices, fold_of = set(), {}
    for g, vs in sorted(by_gender.items()):
        rng.shuffle(vs)
        test_voices |= set(vs[:N_TEST_PER_GENDER])
        for i, v in enumerate(vs[N_TEST_PER_GENDER:]):
            fold_of[v] = i % N_FOLDS  # folds balanced by gender

    rows = []
    for m in meta:
        vg, fold = m["voice_group"], ""
        if vg in bad_voices or m["clip_id"] in bad_clips:
            s = "excluded"
        elif not m["locale"].startswith("ar-"):
            s = "held_foreign"
        elif m["tier"] == "distortion":
            s = "confidence"
        elif vg in test_voices:
            s = "test_voices"
        else:
            s, fold = "train", fold_of[vg]
        rows.append({"clip_id": m["clip_id"], "set": s, "fold": fold, "voice_group": vg,
                     "gender": gender(vg) if m["locale"].startswith("ar-") else ""})

    with open(ROOT / "data" / "splits.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "set", "fold", "voice_group", "gender"])
        w.writeheader()
        w.writerows(rows)

    label = {m["clip_id"]: m["label"] for m in meta}
    print("clips per set:", dict(Counter(r["set"] for r in rows)))
    for s in ("train", "test_voices"):
        print(f"  {s}: {dict(Counter(label[r['clip_id']] for r in rows if r['set'] == s))}, "
              f"{len({r['voice_group'] for r in rows if r['set'] == s})} voices")
    print("test voices:", ", ".join(sorted(test_voices)))
    print("train voices per fold:", dict(sorted(Counter(fold_of.values()).items())))


if __name__ == "__main__":
    main()
