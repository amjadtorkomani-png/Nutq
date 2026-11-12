"""
Check the R in one or more recordings with the trained model (سامع النطق).

The model hears audio only: it does not know which word was asked. It answers one question:
is the R in this word said correctly?

  python scripts/predict.py data/samples/w01_رمل_r.wav data/samples/w19_قمر_gh.wav

Output per file: P(error) after calibration, and a verdict through the confidence gate:
  correct      P(error) <= 0.1
  wrong        P(error) >= 0.9
  close        anything in between -> in the app this goes to the specialist

Input audio: WAV, mono, 16 kHz (other rates are resampled), one word, silence trimmed.
Trained on SYNTHETIC speech only: not validated on real children.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_features import SR, features  # noqa: E402

GATE_LOW, GATE_HIGH = 0.1, 0.9


def load(path):
    y, sr = sf.read(path, dtype="float64")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--model", default=str(HERE.parent / "trained" / "r_baseline.joblib"))
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    bundle = joblib.load(a.model)
    model, calibrator = bundle["model"], bundle["calibrator"]
    for w in a.wavs:
        x = features(load(w))[None, :]
        p = float(calibrator.predict(model.predict_proba(x)[:, 1])[0])
        verdict = "correct" if p <= GATE_LOW else "wrong" if p >= GATE_HIGH else "close (send to specialist)"
        print(f"{Path(w).name}\tP(error)={p:.2f}\t{verdict}")


if __name__ == "__main__":
    main()
