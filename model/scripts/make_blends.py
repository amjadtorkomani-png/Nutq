"""
Make "in-between" R sounds (distortions) with the WORLD vocoder (سامع النطق).

For one voice and one word we have the correct clip C (ر) and an error clip E
(ر replaced by ل / غ / ي / و). Both are said by the same voice, so they differ
mainly where the R is. We:
  1. align C and E with DTW on MFCCs,
  2. find the region where they differ most (the R region),
  3. inside that region only, mix the WORLD spectral envelope and aperiodicity
     of C and E:   log(sp) = (1 - a) * log(sp_C) + a * log(sp_E)
  4. resynthesize.

a = 0.35 and 0.65 give sounds between R and the other letter -> label "distorted".
a = 0 (C resynthesized) and a = 1 (E resynthesized) are also saved, labelled
correct / incorrect, so vocoder artefacts appear in EVERY class and the model
cannot learn "vocoded = error".

Only Arabic voices are used (foreign voices' R is not verified).

Input : data/raw/metadata.csv     Output: data/blend/<engine>/<voice>/<utt_id>.wav
                                          data/blend/metadata.csv
Usage : python scripts/make_blends.py [--max-voices N] [--workers 4]
"""

import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "blend"
SR = 16000
FP = 5.0  # WORLD frame period, ms
HOP = int(SR * FP / 1000)
TARGETS = ("l", "gh", "y", "w")
ALPHAS = (0.35, 0.65)

FIELDS = ["clip_id", "engine", "voice", "voice_group", "locale", "utt_id", "word_id", "position",
          "word_plain", "label", "produced", "produced_char", "tier", "text", "path", "sr",
          "duration_s", "peak", "rms_db", "qc_flag", "review_status", "blend_alpha",
          "blend_region_s", "source_c", "source_e"]


def trim(x, thresh_db=-40.0, pad=0.05):
    fr = int(0.02 * SR)
    e = np.array([np.sqrt(np.mean(x[i:i + fr] ** 2)) for i in range(0, max(1, len(x) - fr), fr)])
    db = 20 * np.log10(e + 1e-9)
    on = np.where(db > db.max() + thresh_db)[0]
    if not len(on):
        return x
    s = max(0, on[0] * fr - int(pad * SR))
    t = min(len(x), (on[-1] + 1) * fr + int(pad * SR))
    return x[s:t]


def world(x):
    f0, t = pw.dio(x, SR, frame_period=FP)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    return f0, sp, ap


def mfcc(x, n):
    m = librosa.feature.mfcc(y=x.astype(np.float32), sr=SR, n_mfcc=20, hop_length=HOP, n_fft=512)
    m = m[:, :n] if m.shape[1] >= n else np.pad(m, ((0, 0), (0, n - m.shape[1])), mode="edge")
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)


HARAKAT = {chr(c) for c in range(0x064B, 0x0653)}


def r_center(text):
    """expected relative position of ر in the word (0..1), from its letter index"""
    letters = [c for c in text if c not in HARAKAT]
    return (letters.index("ر") + 0.5) / len(letters)


def diff_region(mc, me, center):
    """DTW-align C to E; return (start, end) C-frames of the most different region and the C->E map.
    A Gaussian prior around the expected R position keeps the search near the R."""
    _, wp = librosa.sequence.dtw(X=mc, Y=me, metric="euclidean")
    wp = wp[::-1]
    n = mc.shape[1]
    cost = np.zeros(n)
    c2e = np.zeros(n, dtype=int)
    seen = np.zeros(n, dtype=bool)
    for i, j in wp:
        d = np.linalg.norm(mc[:, i] - me[:, j])
        if not seen[i] or d < cost[i]:
            cost[i], c2e[i], seen[i] = d, j, True
    cost = np.convolve(cost, np.ones(5) / 5, mode="same")
    rel = (np.arange(n) + 0.5) / n
    cost = cost * np.exp(-0.5 * ((rel - center) / 0.18) ** 2)
    hot = cost > cost.mean() + 0.75 * cost.std()
    runs, s = [], None
    for i, h in enumerate(np.append(hot, False)):
        if h and s is None:
            s = i
        elif not h and s is not None:
            runs.append((s, i))
            s = None
    if not runs:
        return None, c2e
    s, e = max(runs, key=lambda r: cost[r[0]:r[1]].sum())
    return (max(0, s - 2), min(n, e + 2)), c2e


def synth(f0, sp, ap):
    y = pw.synthesize(f0, np.ascontiguousarray(sp), np.ascontiguousarray(ap), SR, frame_period=FP)
    peak = np.max(np.abs(y)) if len(y) else 1.0
    return (y * (0.9 / peak) if peak > 0.9 else y).astype(np.float32)


def stats(y):
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    return len(y) / SR, float(np.max(np.abs(y))) if len(y) else 0.0, 20 * np.log10(rms + 1e-9)


def process(job):
    c_row, e_row, out_rows = job["c"], job["e"], []
    try:
        xc = trim(sf.read(RAW / c_row["path"], dtype="float64")[0])
        xe = trim(sf.read(RAW / e_row["path"], dtype="float64")[0])
        f0c, spc, apc = world(xc)
        f0e, spe, ape = world(xe)
        region, c2e = diff_region(mfcc(xc, len(f0c)), mfcc(xe, len(f0e)), r_center(c_row["text"]))
        if region is None:
            return [], f"no region {c_row['clip_id']}"
        s, e = region
        w = np.zeros(len(f0c))
        w[s:e] = 1.0
        w = np.convolve(w, np.hanning(7) / np.hanning(7).sum(), mode="same")  # soft edges
        mapped = np.clip(c2e, 0, len(f0e) - 1)

        def save(y, utt_id, label, produced, tier, alpha):
            p = OUT / c_row["engine"] / c_row["voice"] / f"{utt_id}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, y, SR, subtype="PCM_16")
            dur, peak, rms_db = stats(y)
            src = c_row if label == "correct" else e_row
            out_rows.append({
                **{k: c_row[k] for k in ("engine", "voice", "voice_group", "locale", "word_id",
                                         "position", "word_plain")},
                "clip_id": f"{c_row['engine']}__{c_row['voice']}__{utt_id}", "utt_id": utt_id,
                "label": label, "produced": produced,
                "produced_char": src["produced_char"] if alpha in (0.0, 1.0) else f"ر~{e_row['produced_char']}",
                "tier": tier, "text": src["text"], "path": p.relative_to(OUT).as_posix(), "sr": SR,
                "duration_s": f"{dur:.3f}", "peak": f"{peak:.3f}", "rms_db": f"{rms_db:.1f}",
                "qc_flag": "", "review_status": "ok", "blend_alpha": alpha,
                "blend_region_s": f"{s * FP / 1000:.3f}-{e * FP / 1000:.3f}",
                "source_c": c_row["clip_id"], "source_e": e_row["clip_id"]})

        wid, code = c_row["word_id"], e_row["produced"]
        if job["write_c"]:
            save(synth(f0c, spc, apc), f"{wid}_r-voc", "correct", "r", "vocoded", 0.0)
        save(synth(f0e, spe, ape), f"{wid}_{code}-voc", "incorrect", code, "vocoded", 1.0)
        for a in ALPHAS:
            wa = (w * a)[:, None]
            sp = np.exp((1 - wa) * np.log(spc + 1e-12) + wa * np.log(spe[mapped] + 1e-12))
            ap = (1 - wa) * apc + wa * ape[mapped]
            save(synth(f0c, sp, ap), f"{wid}_mix-{code}-{int(a * 100)}", "distorted",
                 f"mix-{code}-{int(a * 100)}", "distortion", a)
        return out_rows, None
    except Exception as ex:  # noqa: BLE001
        return out_rows, f"{c_row['clip_id']} + {e_row['produced']}: {ex}"


def keep_awake():
    """Ask Windows not to idle-sleep while this script runs (like a video player does).
    No setting is changed; the request ends when the process exits."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-voices", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--word-ids", default=None, help="only redo these words (comma list); other rows are kept")
    a = ap.parse_args()
    keep_awake()
    only = set(a.word_ids.split(",")) if a.word_ids else None
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(RAW / "metadata.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("utt_id") and r["locale"].startswith("ar-")]
    by = defaultdict(dict)
    for r in rows:
        by[(r["engine"], r["voice"], r["word_id"])][r["produced"]] = r
    voices = sorted({k[1] for k in by})[: a.max_voices or None]

    jobs = []
    for (eng, voice, wid), d in sorted(by.items()):
        if voice not in voices or "r" not in d or (only and wid not in only):
            continue
        first = True
        for code in TARGETS:
            if code in d:
                jobs.append({"c": d["r"], "e": d[code], "write_c": first})
                first = False

    print(f"{len(jobs)} blend jobs over {len(voices)} voices")
    meta, errors = [], []
    if only and (OUT / "metadata.csv").exists():  # keep the rows of the other words
        with open(OUT / "metadata.csv", encoding="utf-8") as f:
            meta = [r for r in csv.DictReader(f) if r["word_id"] not in only]
    with Pool(a.workers) as pool:
        for i, (out_rows, err) in enumerate(pool.imap_unordered(process, jobs, chunksize=4), 1):
            meta += out_rows
            if err:
                errors.append(err)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(meta, key=lambda m: m["clip_id"]))
    print(f"done: {len(meta)} clips, {len(errors)} problems")
    for e in errors[:10]:
        print("  ", e)


if __name__ == "__main__":
    main()
