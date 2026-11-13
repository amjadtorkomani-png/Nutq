"""
Embeddings from a pretrained multilingual speech model (mHuBERT-147) for the R task.

Why: the DSP model rejects a correct R said a different way (5 voices, 55-86% false rejection).
mHuBERT-147 was trained on 147 languages incl. Arabic and many speakers, so its hidden layers
encode which sound is said more than how the voice sounds.

Per clip (all float16), from the hidden layers of the transformer:
  mean over time of layers 6, 9, 12          -> 3 x 768
  mean of layer 9 in 3 equal time segments   -> 3 x 768  (keeps where in the word a sound is)
  total 4608 numbers

Resumable: clips are written in parts of 2000 to data/features/mhubert/part_XXXX.npz; parts that
exist are skipped. Merge the parts with --merge (writes data/features/mhubert_v1.npz).

The model is read from data/../.hf_cache (downloaded once, kept on D:). No internet needed.

Usage:
  python scripts/extract_embeddings.py --limit 40          (speed test, nothing saved)
  python scripts/extract_embeddings.py                     (all clips of train / test_voices / confidence)
  python scripts/extract_embeddings.py --merge
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

CHILD = ROOT / "data" / "child"
PARTS = ROOT / "data" / "features" / "mhubert"
MODEL_ID = "utter-project/mHuBERT-147"
MEAN_LAYERS = (6, 9, 12)
SEG_LAYER, N_SEG = 9, 3
PART_SIZE = 2000


def keep_awake():
    """Ask Windows not to idle-sleep while this script runs. Ends when the process exits."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def names():
    out = [f"mh_l{l}_mean_{i:03d}" for l in MEAN_LAYERS for i in range(768)]
    out += [f"mh_l{SEG_LAYER}_s{s}_{i:03d}" for s in range(N_SEG) for i in range(768)]
    return out


sys.path.insert(0, str(Path(__file__).resolve().parent))
from listener import embed  # noqa: E402  (one embedding function for training and for the app)


def clip_list():
    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as f:
        keep = {r["clip_id"] for r in csv.DictReader(f) if r["set"] in ("train", "test_voices", "confidence")}
    with open(CHILD / "metadata.csv", encoding="utf-8") as f:
        return [(r["clip_id"], r["path"]) for r in csv.DictReader(f) if r["clip_id"] in keep]


def merge():
    files = sorted(PARTS.glob("part_*.npz"))
    X = np.concatenate([np.load(f)["X"] for f in files])
    ids = np.concatenate([np.load(f)["clip_ids"] for f in files])
    out = ROOT / "data" / "features" / "mhubert_v1.npz"
    np.savez(out, X=X, clip_ids=ids, names=np.array(names()))
    print(f"merged {len(files)} parts: {X.shape} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="speed test on N clips, nothing saved")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.merge:
        return merge()
    keep_awake()

    import torch
    from transformers import HubertModel
    torch.set_num_threads(a.threads)
    model = HubertModel.from_pretrained(MODEL_ID).eval()

    clips = clip_list()
    if a.limit:
        t0 = time.time()
        for cid, path in clips[:: len(clips) // a.limit][: a.limit]:
            e = embed(model, torch, sf.read(CHILD / path, dtype="float32")[0])
        print(f"{a.limit} clips: {(time.time() - t0) / a.limit * 1000:.0f} ms/clip, vector {e.shape}, "
              f"finite {np.isfinite(e.astype(np.float32)).all()}")
        return

    PARTS.mkdir(parents=True, exist_ok=True)
    n_parts = (len(clips) + PART_SIZE - 1) // PART_SIZE
    print(f"{len(clips)} clips in {n_parts} parts of {PART_SIZE}", flush=True)
    t0, done = time.time(), 0
    for k in range(n_parts):
        out = PARTS / f"part_{k:04d}.npz"
        if out.exists():
            continue
        chunk = clips[k * PART_SIZE:(k + 1) * PART_SIZE]
        X = np.zeros((len(chunk), len(names())), dtype=np.float16)
        for i, (cid, path) in enumerate(chunk):
            X[i] = embed(model, torch, sf.read(CHILD / path, dtype="float32")[0])
        np.savez(out.with_suffix(".tmp.npz"), X=X, clip_ids=np.array([c for c, _ in chunk]))
        out.with_suffix(".tmp.npz").replace(out)  # a part is only "done" once fully written
        done += len(chunk)
        rate = done / (time.time() - t0)
        left = sum(1 for j in range(k + 1, n_parts) if not (PARTS / f"part_{j:04d}.npz").exists()) * PART_SIZE
        print(f"  part {k + 1}/{n_parts}  {rate:.1f} clips/s  ~{left / rate / 60:.0f} min left", flush=True)
    print("EMBEDDINGS DONE", flush=True)


if __name__ == "__main__":
    main()
