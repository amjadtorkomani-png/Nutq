"""
Train the سامع النطق model for R and measure it.

Input : data/features/mhubert_v1.npz   (4608 mHuBERT-147 numbers per clip, from extract_embeddings.py)
        data/child/metadata.csv, data/splits.csv
Output: models/nutq_r.joblib            (r_model + calibrator + word_model + letter_model + thresholds)
        reports/model_metrics.json

Three logistic regressions on the same embedding (StandardScaler first):
  r_model  correct / incorrect R, class-balanced, C=0.05, all train clips
           calibrated with isotonic regression on 5-fold out-of-fold predictions (folds = voices)
  word     25 words (error versions count as their word)       up to 28 clips per utterance, C=0.02
  letter   the sound in the R place (21 classes, "del" = none)  up to 28 clips per utterance, C=0.02

Everything is measured on the 6 test voices that no model saw, plus the recordings in data/real/
(when there are any): python scripts/train_model.py
"""

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import listener  # noqa: E402

ROOT = listener.ROOT
THRESHOLDS = {"lo": 0.1, "hi": 0.9, "word_min": 0.25, "letter_min": 0.5}
PER_UTT = 28
SEED = 0


def lr(C, balanced=False, it=600):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=it,
                                                              class_weight="balanced" if balanced else None))


def share(v):
    c = Counter(v)
    n = max(1, len(v))
    return {k: round(c[k] / n, 4) for k in ("correct", "close", "incorrect")}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    log = lambda s: print(f"{s}  ({time.time() - t0:.0f}s)", flush=True)
    meta = {r["clip_id"]: r for r in csv.DictReader(open(ROOT / "data" / "child" / "metadata.csv", encoding="utf-8"))}
    split = {r["clip_id"]: r for r in csv.DictReader(open(ROOT / "data" / "splits.csv", encoding="utf-8"))}
    e = np.load(ROOT / "data" / "features" / "mhubert_v1.npz")
    all_ids = e["clip_ids"]
    keep = np.array([split.get(c, {}).get("set") in ("train", "test_voices") for c in all_ids])
    E = e["X"][keep]  # float16
    ids = all_ids[keep]
    del e
    col = lambda k, src=meta: np.array([src[i][k] for i in ids])
    sets, folds = col("set", split), col("fold", split)
    word, produced, label, alpha = col("word_id"), col("produced"), col("label"), col("blend_alpha")
    y = (label == "incorrect").astype(int)
    tr, te = sets == "train", sets == "test_voices"
    f32 = lambda m: E[m].astype(np.float32)
    log(f"{tr.sum()} train clips, {te.sum()} test clips")

    # 1. R model + calibration
    oof = np.zeros(len(y))
    for k in sorted(set(folds[tr])):
        va, fit = tr & (folds == k), tr & (folds != k)
        oof[va] = lr(0.05, True).fit(f32(fit), y[fit]).predict_proba(f32(va))[:, 1]
    r_model = lr(0.05, True).fit(f32(tr), y[tr])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[tr], y[tr])
    Xte = f32(te)
    p_raw = r_model.predict_proba(Xte)[:, 1]
    p_r = iso.predict(p_raw)
    log("r_model done")

    # 2. word + letter models (balanced sample: up to PER_UTT clips of every utterance, no in-between blends)
    rng = np.random.default_rng(SEED)
    by_utt = defaultdict(list)
    for i in np.where(tr & (alpha == ""))[0]:
        by_utt[meta[ids[i]]["utt_id"]].append(i)
    sub = np.sort(np.concatenate([rng.choice(v, min(PER_UTT, len(v)), replace=False) for v in by_utt.values()]))
    Xs = E[sub].astype(np.float32)
    word_model = lr(0.02, it=300).fit(Xs, word[sub])
    letter_model = lr(0.02, it=300).fit(Xs, produced[sub])
    del Xs
    Pw, Pl = word_model.predict_proba(Xte), letter_model.predict_proba(Xte)
    log("word + letter done")

    bundle = {"r_model": r_model, "calibrator": iso, "word_model": word_model, "letter_model": letter_model,
              "thresholds": THRESHOLDS, "embedding_model": listener.EMBEDDING_MODEL,
              "embedding_layout": "mean of layers 6, 9, 12 + layer 9 in 3 time segments (4608)",
              "child_voice": listener.CHILD}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, listener.MODEL_FILE)

    # 3. measure on the test voices
    yt, wt, pt = y[te], word[te], produced[te]
    M = {"thresholds": THRESHOLDS, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    pred = (p_r >= 0.5).astype(int)
    M["r_model"] = {"auc": float(roc_auc_score(yt, p_raw)),
                    "balanced_acc": float((np.mean(pred[yt == 0] == 0) + np.mean(pred[yt == 1] == 1)) / 2),
                    "false_rejection": float(np.mean(pred[yt == 0] == 1)), "missed_errors": float(np.mean(pred[yt == 1] == 0))}
    wpred, lpred = word_model.classes_[Pw.argmax(1)], letter_model.classes_[Pl.argmax(1)]
    M["word_acc"] = float(np.mean(wpred == wt))
    M["letter_acc"] = float(np.mean(lpred == pt))
    M["letter_acc_errors_only"] = float(np.mean(lpred[yt == 1] == pt[yt == 1]))

    verdict = np.array([listener.decide(bundle, p_r[i], Pw[i], Pl[i], wt[i])["verdict"] for i in range(len(yt))])
    reason = np.array([listener.decide(bundle, p_r[i], Pw[i], Pl[i], wt[i])["reason"] for i in range(len(yt))])
    M["system_said_correct"] = share(verdict[yt == 0])
    M["system_said_incorrect"] = share(verdict[yt == 1])
    M["system_reasons_on_correct_words"] = dict(Counter(reason[(yt == 0) & (verdict == "incorrect")]))
    words = list(word_model.classes_)
    other = np.array([words[(words.index(w) + 1 + rng.integers(len(words) - 1)) % len(words)] for w in wt])
    sw = [listener.decide(bundle, p_r[i], Pw[i], Pl[i], other[i])["verdict"] for i in np.where(yt == 0)[0]]
    M["system_other_word"] = share(sw)
    log("test voices measured")

    # 4. real recordings (data/real/log.csv), if any
    real = ROOT / "data" / "real" / "log.csv"
    if real.exists():
        rows = [r for r in csv.DictReader(open(real, encoding="utf-8")) if r["mode"] != "test" and r["expected"]]
        if rows:
            del E, Xte
            L = listener.Listener()
            out = defaultdict(list)
            for r in rows:
                res = L.judge(str(ROOT / "data" / "real" / r["file"]), r["word_id"], adult=r.get("adult", "1") == "1")
                out[r["expected"]].append(res["verdict"])
            M["real"] = {"n": len(rows), "said_correct": share(out["correct"]), "n_correct": len(out["correct"]),
                         "said_incorrect": share(out["incorrect"]), "n_incorrect": len(out["incorrect"])}
            log(f"{len(rows)} real recordings measured")

    (ROOT / "reports" / "model_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(M, ensure_ascii=False, indent=1))
    print("MODEL DONE", flush=True)


if __name__ == "__main__":
    main()
