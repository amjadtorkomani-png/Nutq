"""
Turn adult TTS clips into child-like clips (سامع النطق).

For every source voice we create K "virtual children". Each virtual child has
its own fixed settings, applied with Praat's "Change gender":
  formant_shift   1.12 - 1.28   (shorter vocal tract -> higher formants)
  pitch_median    230 - 320 Hz  (children's F0 range)
  pitch_range     0.8 - 1.4     (flat or lively intonation)
  duration        1.00 - 1.25   (children often speak slower)

Silence at the start and end is trimmed (50 ms kept on each side).

Input : data/raw/metadata.csv (generate_tts.py) and data/blend/metadata.csv (make_blends.py)
Output: data/child/<engine>__<voice>__c<k>/<utt_id>.wav and data/child/metadata.csv

voice_group stays the SOURCE voice: all virtual children made from one TTS voice
must stay on the same side of the train/validation split.

Usage:
  python scripts/childify.py --children-per-voice 4
"""

import argparse
import csv
import hashlib
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BLEND = ROOT / "data" / "blend"
OUT = ROOT / "data" / "child"
SR = 16000

RANGES = {
    "formant_shift": (1.12, 1.28),  # above ~1.3 some clips sounded damaged
    "pitch_median": (230.0, 320.0),
    "pitch_range": (0.8, 1.4),
    "duration": (1.00, 1.25),
}


def child_params(engine, voice, k):
    seed = int(hashlib.md5(f"{engine}|{voice}|{k}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return {name: round(rng.uniform(lo, hi), 3) for name, (lo, hi) in RANGES.items()}


def trim(audio, sr, thresh_db=-40.0, pad_s=0.05):
    fr = int(0.02 * sr)
    if len(audio) < fr:
        return audio
    frames = len(audio) // fr
    e = np.array([np.sqrt(np.mean(audio[i * fr:(i + 1) * fr] ** 2)) for i in range(frames)])
    e_db = 20 * np.log10(e + 1e-9)
    on = np.where(e_db > e_db.max() + thresh_db)[0]  # relative to the loudest frame
    if len(on) == 0:
        return audio
    pad = int(pad_s * sr)
    s = max(0, on[0] * fr - pad)
    t = min(len(audio), (on[-1] + 1) * fr + pad)
    return audio[s:t]


def process(job):
    src, dst, p = job
    try:
        audio, sr = sf.read(src, dtype="float64")
        audio = trim(audio, sr)
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        out = call(snd, "Change gender", 75, 600, p["formant_shift"], p["pitch_median"],
                   p["pitch_range"], p["duration"])
        y = out.values[0].astype(np.float32)
        if out.sampling_frequency != SR:
            from scipy.signal import resample_poly
            g = np.gcd(int(out.sampling_frequency), SR)
            y = resample_poly(y, SR // g, int(out.sampling_frequency) // g).astype(np.float32)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 0.95:
            y = y * (0.95 / peak)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, y, SR, subtype="PCM_16")
        return dst, len(y) / SR, None
    except Exception as e:  # noqa: BLE001
        return dst, 0.0, str(e)


def keep_awake():
    """Ask Windows not to idle-sleep while this script runs (like a video player does).
    No setting is changed; the request ends when the process exits."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--children-per-voice", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--word-ids", default=None, help="re-make clips of these words only (comma list)")
    a = ap.parse_args()
    keep_awake()
    only = set(a.word_ids.split(",")) if a.word_ids else None
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    src_rows = []
    for base in (RAW, BLEND):
        if (base / "metadata.csv").exists():
            with open(base / "metadata.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("utt_id"):
                        r["_base"] = base
                        r.setdefault("review_status", "ok")
                        r.setdefault("blend_alpha", "")
                        src_rows.append(r)

    jobs, meta = [], []
    for r in src_rows:
        for k in range(a.children_per_voice):
            p = child_params(r["engine"], r["voice"], k)
            child_voice = f"{r['engine']}__{r['voice']}__c{k}"
            dst = OUT / child_voice / f"{r['utt_id']}.wav"
            row = {**{c: r[c] for c in ("engine", "voice", "voice_group", "locale", "utt_id", "word_id",
                                        "position", "word_plain", "label", "produced", "produced_char",
                                        "tier", "text", "review_status", "blend_alpha")},
                   "clip_id": f"{child_voice}__{r['utt_id']}", "child_voice": child_voice,
                   "source_clip": r["clip_id"], "path": dst.relative_to(OUT).as_posix(), **p}
            meta.append(row)
            if (only and r["word_id"] in only) or (not only and (a.overwrite or not dst.exists())):
                jobs.append((r["_base"] / r["path"], dst, p))

    print(f"{len(src_rows)} source clips -> {len(meta)} child clips ({len(jobs)} to make)")
    durations, errors = {}, []
    with Pool(a.workers) as pool:
        for i, (dst, dur, err) in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            durations[dst] = dur
            if err:
                errors.append(f"{dst.name}: {err}")
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}")

    # Durations: from this run, else from the previous metadata.csv. Reading every file on a
    # slow disk just for its duration took close to an hour, so unknown ones stay empty and
    # check_child.py (which reads every file anyway) fills them in.
    cache = {}
    if (OUT / "metadata.csv").exists():
        with open(OUT / "metadata.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("duration_s") and float(r["duration_s"]) > 0:
                    cache[r["clip_id"]] = r["duration_s"]
    for m in meta:
        d = durations.get(OUT / m["path"])
        m["duration_s"] = f"{d:.3f}" if d else cache.get(m["clip_id"], "")

    fields = ["clip_id", "child_voice", "voice_group", "engine", "voice", "locale", "utt_id",
              "word_id", "position", "word_plain", "label", "produced", "produced_char", "tier",
              "text", "path", "duration_s", "formant_shift", "pitch_median", "pitch_range",
              "duration", "review_status", "blend_alpha", "source_clip"]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(meta, key=lambda m: m["clip_id"]))
    print(f"done: {len(jobs) - len(errors)} made, {len(errors)} failed")
    for e in errors[:10]:
        print("  FAIL", e)


if __name__ == "__main__":
    main()
