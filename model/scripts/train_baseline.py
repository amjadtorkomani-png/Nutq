"""
Train and evaluate the first R model (سامع النطق): is the R in this word correct or not?

Input : data/features/dsp_v1.npz, data/child/metadata.csv, data/splits.csv
Output: models/r_baseline.joblib, reports/baseline_report.md, reports/baseline_metrics.json

Positive class = error (label "incorrect"). What is measured:
  1. 5-fold cross-validation, split by voice (out-of-fold predictions)
  2. test on 6 voices never seen in training
  3. unseen substitutions: some replacement letters are removed from training, then tested
  4. confidence test on "in-between" R clips of the test voices (not trained on)
  5. which feature family carries the result
All data here is synthetic (TTS + child-voice conversion). These numbers say nothing yet
about real children.

Usage: python scripts/train_baseline.py
"""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
UNSEEN = ("z", "q", "m", "ain", "kh")  # replacement letters hidden from training in experiment 3
GATES = ((0.5, 0.5), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9))


def hgb(n_iter=400):
    return HistGradientBoostingClassifier(max_iter=n_iter, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=400, class_weight="balanced"))


def metrics(y, p, thr=0.5):
    pred = p >= thr
    tp, fn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 1))
    tn, fp = np.sum(~pred & (y == 0)), np.sum(pred & (y == 0))
    rec_err, rec_ok = tp / max(1, tp + fn), tn / max(1, tn + fp)
    return {"n": int(len(y)), "balanced_acc": float((rec_err + rec_ok) / 2),
            "error_recall": float(rec_err), "false_rejection": float(fp / max(1, tn + fp)),
            "missed_errors": float(fn / max(1, tp + fn)), "auc": float(roc_auc_score(y, p))}


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(out)


def gate_table(y, p):
    rows = []
    for lo, hi in GATES:
        decided = (p <= lo) | (p >= hi)
        d_y, d_pred = y[decided], p[decided] >= hi
        rows.append({"lo": lo, "hi": hi, "referred": float(1 - decided.mean()),
                     "false_rejection": float(np.sum(d_pred & (d_y == 0)) / max(1, np.sum(d_y == 0))),
                     "missed_errors": float(np.sum(~d_pred & (d_y == 1)) / max(1, np.sum(d_y == 1)))})
    return rows


def by_group(keys, y, p, positive_only=False):
    out = {}
    for k in sorted(set(keys)):
        m = keys == k
        if positive_only:
            out[k] = {"n": int(m.sum()), "recall": float(np.mean(p[m] >= 0.5))}
        elif len(set(y[m])) == 2:
            out[k] = metrics(y[m], p[m])
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    d = np.load(ROOT / "data" / "features" / "dsp_v1.npz")
    X, ids, names = d["X"], d["clip_ids"], [str(n) for n in d["names"]]
    with open(ROOT / "data" / "child" / "metadata.csv", encoding="utf-8") as f:
        meta = {r["clip_id"]: r for r in csv.DictReader(f)}
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as f:
        split = {r["clip_id"]: r for r in csv.DictReader(f)}

    col = lambda k, src=meta: np.array([src[i][k] for i in ids])
    sets, folds, voices = col("set", split), col("fold", split), col("voice_group")
    label, produced, tier, position, word = col("label"), col("produced"), col("tier"), col("position"), col("word_id")
    y = (label == "incorrect").astype(int)
    tr, te, cf = sets == "train", sets == "test_voices", sets == "confidence"
    print(f"train {tr.sum()}, test_voices {te.sum()}, confidence {cf.sum()}, features {X.shape[1]}", flush=True)
    R = {"n_features": X.shape[1], "n_train": int(tr.sum()), "n_test": int(te.sum())}

    # ---- 1) cross-validation by voice
    oof = {"hgb": np.zeros(len(y)), "logreg": np.zeros(len(y))}
    for k in sorted(set(folds[tr])):
        va = tr & (folds == k)
        fit = tr & (folds != k)
        for name, make in (("hgb", hgb), ("logreg", logreg)):
            oof[name][va] = make().fit(X[fit], y[fit]).predict_proba(X[va])[:, 1]
        print(f"  fold {k}: hgb bal-acc {metrics(y[va], oof['hgb'][va])['balanced_acc']:.3f}, "
              f"logreg {metrics(y[va], oof['logreg'][va])['balanced_acc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    R["cv"] = {n: metrics(y[tr], oof[n][tr]) for n in oof}
    R["cv_by_voice"] = by_group(voices[tr], y[tr], oof["hgb"][tr])

    # ---- 2) final model + calibration, test on unseen voices
    model = hgb().fit(X[tr], y[tr])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof["hgb"][tr], y[tr])
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = iso.predict(p_raw)
    base = logreg().fit(X[tr], y[tr])
    R["test"] = {"hgb": metrics(y[te], p_raw[te]), "logreg": metrics(y[te], base.predict_proba(X[te])[:, 1])}
    R["test_ece"] = {"raw": ece(y[te], p_raw[te]), "calibrated": ece(y[te], p_cal[te])}
    R["test_gate"] = gate_table(y[te], p_cal[te])
    R["test_by_position"] = by_group(position[te], y[te], p_raw[te])
    R["test_by_word"] = by_group(word[te], y[te], p_raw[te])
    R["test_by_voice"] = by_group(voices[te], y[te], p_raw[te])
    err = te & (y == 1)
    R["test_recall_by_produced"] = by_group(produced[err], y[err], p_raw[err], positive_only=True)
    ok = te & (y == 0)
    R["test_false_rejection_by_tier"] = {t: float(np.mean(p_raw[ok & (tier == t)] >= 0.5)) for t in ("correct", "vocoded")}
    print(f"  test voices: bal-acc {R['test']['hgb']['balanced_acc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- 3) unseen substitutions
    hide = np.isin(produced, UNSEEN)
    m3 = hgb().fit(X[tr & ~hide], y[tr & ~hide])
    p3 = m3.predict_proba(X)[:, 1]
    R["unseen_letters"] = {"letters": list(UNSEEN),
                           "recall_unseen": float(np.mean(p3[err & hide] >= 0.5)),
                           "recall_seen": float(np.mean(p3[err & ~hide] >= 0.5)),
                           "recall_same_letters_when_trained_on": float(np.mean(p_raw[err & hide] >= 0.5))}
    print(f"  unseen letters recall {R['unseen_letters']['recall_unseen']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- 4) confidence on in-between clips (test voices only)
    test_voice_set = set(voices[te])
    cft = cf & np.isin(voices, list(test_voice_set))
    alpha = np.array([meta[i]["blend_alpha"] or "" for i in ids])
    target = np.array([p.split("-")[1] if p.startswith("mix-") else "" for p in produced])
    conf = defaultdict(dict)
    for t in ("l", "gh", "y", "w"):
        voc_ok = te & (tier == "vocoded") & (y == 0)
        voc_err = te & (tier == "vocoded") & (produced == t)
        steps = [("0 (ر)", voc_ok), ("0.35", cft & (target == t) & (alpha == "0.35")),
                 ("0.65", cft & (target == t) & (alpha == "0.65")), ("1", voc_err)]
        for name, m in steps:
            if m.any():
                conf[t][name] = {"n": int(m.sum()), "mean_p_error": float(p_cal[m].mean()),
                                 "unsure_0.2_0.8": float(np.mean((p_cal[m] > 0.2) & (p_cal[m] < 0.8)))}
    R["confidence"] = conf

    # ---- 5) feature families (each family alone, train -> test voices)
    fam = defaultdict(list)
    for i, n in enumerate(names):
        fam[n.split("_")[0].rstrip("0123456789").replace("ddmfcc", "mfcc").replace("dmfcc", "mfcc")].append(i)
    R["families_alone"] = {}
    for f, cols in fam.items():
        if len(cols) < 2:
            continue
        pf = hgb(150).fit(X[tr][:, cols], y[tr]).predict_proba(X[te][:, cols])[:, 1]
        R["families_alone"][f] = {"n_features": len(cols), "balanced_acc": metrics(y[te], pf)["balanced_acc"]}
    print(f"  families done ({time.time() - t0:.0f}s)", flush=True)

    # ---- save
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump({"model": model, "calibrator": iso, "feature_names": names, "positive_class": "incorrect"},
                ROOT / "models" / "r_baseline.joblib")
    (ROOT / "reports" / "baseline_metrics.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")

    pct = lambda v: f"{100 * v:.1f}%"
    L = ["# سامع النطق: نتائج الموديل الأول (حرف الراء)", "",
         "> كل البيانات هنا اصطناعية (TTS + تحويل لصوت طفل). هذي الأرقام لا تقول شيئًا بعد عن أطفال حقيقيين.", "",
         f"- الميزات: {X.shape[1]} (DSP). التدريب: {tr.sum()} مقطع من 23 صوت. الاختبار: {te.sum()} مقطع من 6 أصوات ما شافها الموديل.",
         "- الموديل: HistGradientBoosting. خط الأساس: Logistic Regression.", "",
         "## 1) النتيجة العامة", "",
         "| القياس | CV (حسب الصوت) | أصوات الاختبار | خط الأساس (اختبار) |", "|---|---|---|---|"]
    for key, title in (("balanced_acc", "Balanced accuracy"), ("error_recall", "أخطاء انكشفت (recall)"),
                       ("false_rejection", "صحيح انحسب غلط"), ("auc", "AUC")):
        L.append(f"| {title} | {pct(R['cv']['hgb'][key])} | {pct(R['test']['hgb'][key])} | {pct(R['test']['logreg'][key])} |")
    L += ["", f"ECE (صدق نسبة الثقة) على الاختبار: قبل المعايرة {R['test_ece']['raw']:.3f}، بعدها {R['test_ece']['calibrated']:.3f}.", "",
          "## 2) بوابة الثقة (على أصوات الاختبار)", "",
          "| يحكم \"صح\" إذا P(غلط) ≤ | يحكم \"غلط\" إذا ≥ | يتحول للأخصائي | صحيح انحسب غلط | أخطاء فاتت |", "|---|---|---|---|---|"]
    L += [f"| {g['lo']} | {g['hi']} | {pct(g['referred'])} | {pct(g['false_rejection'])} | {pct(g['missed_errors'])} |" for g in R["test_gate"]]
    L += ["", "## 3) حسب موضع الراء", "", "| الموضع | Balanced accuracy | أخطاء انكشفت | صحيح انحسب غلط |", "|---|---|---|---|"]
    L += [f"| {k} | {pct(v['balanced_acc'])} | {pct(v['error_recall'])} | {pct(v['false_rejection'])} |" for k, v in R["test_by_position"].items()]
    L += ["", "## 4) نسبة كشف كل نوع غلط (أصوات الاختبار)", "", "| بدل الراء | العدد | انكشف |", "|---|---|---|"]
    L += [f"| {k} | {v['n']} | {pct(v['recall'])} |" for k, v in sorted(R["test_recall_by_produced"].items(), key=lambda kv: kv[1]["recall"])]
    u = R["unseen_letters"]
    L += ["", "## 5) حروف ما تدرب عليها الموديل", "",
          f"شلنا من التدريب الحروف: {', '.join(u['letters'])}. نسبة كشفها: **{pct(u['recall_unseen'])}** "
          f"(ولما تدرب عليها: {pct(u['recall_same_letters_when_trained_on'])}؛ وباقي الحروف: {pct(u['recall_seen'])}).", "",
          "## 6) اختبار الثقة على \"الراء بين بين\" (أصوات الاختبار، ما تدرب عليها)", "",
          "| نحو الحرف | نسبة الخلط | العدد | متوسط P(غلط) | في منطقة \"مو متأكد\" |", "|---|---|---|---|---|"]
    for t, steps in R["confidence"].items():
        L += [f"| {t} | {a} | {v['n']} | {v['mean_p_error']:.2f} | {pct(v['unsure_0.2_0.8'])} |" for a, v in steps.items()]
    L += ["", "## 7) كل عائلة ميزات لحالها", "", "| العائلة | عدد الميزات | Balanced accuracy |", "|---|---|---|"]
    L += [f"| {k} | {v['n_features']} | {pct(v['balanced_acc'])} |" for k, v in sorted(R["families_alone"].items(), key=lambda kv: -kv[1]["balanced_acc"])]
    words_sorted = sorted(R["test_by_word"].items(), key=lambda kv: kv[1]["balanced_acc"])
    L += ["", "## 8) أضعف 5 كلمات", "", "| الكلمة | Balanced accuracy | صحيح انحسب غلط |", "|---|---|---|"]
    wname = {m["word_id"]: m["word_plain"] for m in meta.values()}
    L += [f"| {wname[k]} | {pct(v['balanced_acc'])} | {pct(v['false_rejection'])} |" for k, v in words_sorted[:5]]
    L += ["", f"صحيح انحسب غلط حسب النوع: عادي {pct(R['test_false_rejection_by_tier']['correct'])}، "
              f"vocoder {pct(R['test_false_rejection_by_tier']['vocoded'])}."]
    (ROOT / "reports" / "baseline_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"saved model and report ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
