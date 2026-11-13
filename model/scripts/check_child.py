"""
Check the child-like clips after childify.py (سامع النطق).

For every clip in data/child/metadata.csv:
  - the file opens and is not too short          -> else flag "broken" / "too_short"
  - Praat finds voiced frames                     -> else flag "unvoiced"
  - median F0 is in a child-like range (180-400)  -> else flag "f0_out_of_range"
Broken clips are re-made by running childify.py again after deleting them (--fix).

Writes data/child/check.csv (one row per flagged clip) and prints a summary.
Usage: python scripts/check_child.py [--fix] [--workers 4]
"""

import argparse
import csv
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "data" / "child"


def check(row):
    """returns (clip_id, duration_s or None, flag-tuple or None)"""
    cid = row["clip_id"]
    r = _check(row)
    return cid, r[0], r[1]


def _check(row):
    p = CHILD / row["path"]
    try:
        y, sr = sf.read(p, dtype="float64")
    except Exception as e:  # noqa: BLE001
        return None, (row["clip_id"], row["path"], "broken", str(e)[:80])
    dur = len(y) / sr
    if len(y) < 0.2 * sr:
        return dur, (row["clip_id"], row["path"], "too_short", f"{dur:.2f}s")
    pitch = parselmouth.Sound(y, sampling_frequency=sr).to_pitch(pitch_floor=75, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) == 0:
        return dur, (row["clip_id"], row["path"], "unvoiced", "")
    med = float(np.median(f0))
    if not 180 <= med <= 400:
        return dur, (row["clip_id"], row["path"], "f0_out_of_range", f"{med:.0f} Hz")
    return dur, None


def keep_awake():
    """Ask Windows not to idle-sleep while this script runs (like a video player does).
    No setting is changed; the request ends when the process exits."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fix", action="store_true", help="delete broken / too_short files so childify.py re-makes them")
    ap.add_argument("--recheck", action="store_true",
                    help="only check the clips flagged broken/too_short in the last check.csv (after they were re-made)")
    a = ap.parse_args()
    keep_awake()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(CHILD / "metadata.csv", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    fields = list(all_rows[0].keys())
    rows = all_rows
    previous = []
    if a.recheck:
        with open(CHILD / "check.csv", encoding="utf-8") as f:
            previous = list(csv.DictReader(f))
        redo = {r["clip_id"] for r in previous if r["flag"] in ("broken", "too_short")}
        rows = [r for r in all_rows if r["clip_id"] in redo]
        print(f"recheck: {len(rows)} re-made clips")

    flagged, durs = [], {}
    with Pool(a.workers) as pool:
        for i, (cid, dur, r) in enumerate(pool.imap_unordered(check, rows, chunksize=64), 1):
            if dur:
                durs[cid] = f"{dur:.3f}"
            if r:
                flagged.append(r)
            if i % 10000 == 0:
                print(f"  {i}/{len(rows)}", flush=True)

    # write the measured durations back into metadata.csv
    for r in all_rows:
        if r["clip_id"] in durs:
            r["duration_s"] = durs[r["clip_id"]]
    with open(CHILD / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    if a.recheck:  # keep the earlier non-broken flags (unvoiced etc.) in check.csv
        flagged += [(r["clip_id"], r["path"], r["flag"], r["detail"]) for r in previous
                    if r["flag"] not in ("broken", "too_short")]
    with open(CHILD / "check.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "path", "flag", "detail"])
        w.writerows(sorted(flagged))

    by_flag = Counter(r[2] for r in flagged)
    print(f"{len(rows)} clips checked, {len(flagged)} flagged in check.csv: {dict(by_flag)}")
    meta = {r["clip_id"]: r for r in all_rows}
    for flag in by_flag:
        ex = [r for r in flagged if r[2] == flag]
        tiers = Counter(meta[r[0]]["tier"] for r in ex)
        prods = Counter(meta[r[0]]["produced"] for r in ex).most_common(5)
        print(f"  {flag}: by tier {dict(tiers)}, top produced {prods}")
    if a.fix:
        n = 0
        for cid, path, flag, _ in flagged:
            if flag in ("broken", "too_short"):
                (CHILD / path).unlink(missing_ok=True)
                n += 1
        print(f"deleted {n} files; run childify.py again to re-make them")


if __name__ == "__main__":
    main()
