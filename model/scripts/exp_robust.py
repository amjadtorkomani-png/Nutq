"""
Experiment: make the DSP features less sensitive to the voice (سامع النطق).

Finding from the first model: 5 voices have a correct R (checked by ear) but 55-86% of their
correct clips are rejected. Their features differ in overall timbre, not in the R itself.
Here the same model is trained on feature variants that remove static voice colour:

  base        features as extracted
  cmn         melgrid and per-segment spectral features: the mean over the 6 time segments is
              subtracted per band (keeps how the sound changes in time, drops the voice's colour)
  cmn_nomean  cmn, and the remaining whole-word averages (mfcc means, spec means, F0, HNR) dropped

Usage: python scripts/exp_robust.py
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
N_SEG = 6


def variant(X, names, kind):
    X = X.copy()
    keep = np.ones(len(names), dtype=bool)
    if kind == "base":
        return X, keep
    groups = {}
    for i, n in enumerate(names):
        if n.startswith("melgrid_s"):
            groups.setdefault("mel_" + n.split("_")[2], []).append(i)
        elif n.startswith("spec_") and n.rsplit("_", 1)[1].startswith("s") and n.rsplit("_", 1)[1][1:].isdigit():
            groups.setdefault(n.rsplit("_", 1)[0], []).append(i)
        elif n.startswith("voice_hnr_s"):
            groups.setdefault("hnr", []).append(i)
    for cols in groups.values():
        assert len(cols) == N_SEG
        X[:, cols] -= X[:, cols].mean(axis=1, keepdims=True)
    if kind == "cmn_nomean":
        for i, n in enumerate(names):
            if (n.startswith("mfcc") and n.endswith("_mean")) or (n.startswith("spec_") and n.endswith("_mean")) \
                    or n in ("voice_f0_median", "voice_hnr_mean"):
                keep[i] = False
    return X, keep


def rates(y, p):
    pred = p >= 0.5
    fr = np.sum(pred & (y == 0)) / max(1, np.sum(y == 0))
    miss = np.sum(~pred & (y == 1)) / max(1, np.sum(y == 1))
    return 1 - (fr + miss) / 2, fr, miss


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    d = np.load(ROOT / "data" / "features" / "dsp_v1.npz")
    X0, ids, names = d["X"], d["clip_ids"], [str(n) for n in d["names"]]
    meta = {r["clip_id"]: r for r in csv.DictReader(open(ROOT / "data" / "child" / "metadata.csv", encoding="utf-8"))}
    split = {r["clip_id"]: r for r in csv.DictReader(open(ROOT / "data" / "splits.csv", encoding="utf-8"))}
    sets = np.array([split[i]["set"] for i in ids])
    folds = np.array([split[i]["fold"] for i in ids])
    voices = np.array([meta[i]["voice_group"] for i in ids])
    y = np.array([meta[i]["label"] == "incorrect" for i in ids]).astype(int)
    tr, te = sets == "train", sets == "test_voices"
    t0 = time.time()
    for kind in ("base", "cmn", "cmn_nomean"):
        X, keep = variant(X0, names, kind)
        X = X[:, keep]
        mk = lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                                                    l2_regularization=1.0, class_weight="balanced", random_state=0)
        oof = np.zeros(len(y))
        for k in sorted(set(folds[tr])):
            va, fit = tr & (folds == k), tr & (folds != k)
            oof[va] = mk().fit(X[fit], y[fit]).predict_proba(X[va])[:, 1]
        p_all = mk().fit(X[tr], y[tr]).predict_proba(X)[:, 1]
        cv, tst = rates(y[tr], oof[tr]), rates(y[te], p_all[te])
        per_voice = sorted(((rates(y[tr & (voices == v)], oof[tr & (voices == v)])[1], v) for v in set(voices[tr])), reverse=True)
        per_voice += sorted(((rates(y[te & (voices == v)], p_all[te & (voices == v)])[1], v) for v in set(voices[te])), reverse=True)
        n_bad = sum(fr > 0.2 for fr, _ in per_voice)
        print(f"{kind:11s} ({keep.sum()} feats)  CV bal {cv[0]:.3f} falseRej {cv[1]:.3f} miss {cv[2]:.3f} | "
              f"TEST bal {tst[0]:.3f} falseRej {tst[1]:.3f} miss {tst[2]:.3f} | voices with falseRej>20%: {n_bad}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        print("   worst: " + ", ".join(f"{v.split('-')[2].replace('Neural', '')} {fr:.0%}" for fr, v in sorted(per_voice, reverse=True)[:6]), flush=True)


if __name__ == "__main__":
    main()
