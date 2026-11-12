"""
Feature extraction for the R task (سامع النطق): one fixed-length vector per clip.

Input : data/child/metadata.csv + the WAV files (16 kHz mono)
Output: data/features/dsp_v1.npz  (X float32 [n_clips, n_features], clip_ids, names)

The model gets the audio only (no word, no position), so every feature family is computed
over the whole word AND over 6 equal time segments: the R can be at the start, middle or end.

Feature families
  melgrid_*   mean log-mel energy, 24 bands x 6 time segments (a coarse picture of the word)
  mfcc_*      13 MFCC + delta + delta-delta: mean and std over the word
  spec_*      spectral centroid / spread / rolloff / flatness, ZCR, frame energy
  mod_*       amplitude modulation of the 300-3000 Hz envelope (tongue trill = 15-35 Hz ripple)
  voice_*     F0, voicing ratio, harmonics-to-noise ratio (HNR)
  dur_s       clip duration

Usage:
  python scripts/extract_features.py --limit 400      (speed test)
  python scripts/extract_features.py --workers 4
"""

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "data" / "child"
OUT = ROOT / "data" / "features"
SR, N_FFT, HOP, N_MELS, N_SEG, N_MFCC = 16000, 400, 160, 24, 6, 13
ENV_SR = 200

_MEL = None


def mel_fb():
    global _MEL
    if _MEL is None:
        import librosa
        _MEL = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=50, fmax=8000).astype(np.float64)
    return _MEL


def seg_means(x, n=N_SEG):
    """mean of x (frames on axis 0) in n equal time segments"""
    parts = np.array_split(np.arange(len(x)), n)
    return np.stack([x[p].mean(axis=0) if len(p) else np.zeros(x.shape[1:]) for p in parts])


def names():
    out = [f"melgrid_s{s}_b{b:02d}" for s in range(N_SEG) for b in range(N_MELS)]
    for kind in ("mfcc", "dmfcc", "ddmfcc"):
        out += [f"{kind}{i:02d}_{st}" for st in ("mean", "std") for i in range(N_MFCC)]
    desc = ("centroid", "spread", "rolloff", "flatness", "zcr", "energy")
    out += [f"spec_{d}_{st}" for d in desc for st in ("mean", "std")]
    out += [f"spec_{d}_s{s}" for d in ("centroid", "flatness", "zcr", "energy") for s in range(N_SEG)]
    out += ["mod_frac_2_8", "mod_frac_8_15", "mod_frac_15_35", "mod_frac_35_60",
            "mod_trill_max", "mod_trill_pos", "mod_trill_mean", "mod_peaks", "mod_peaks_per_s",
            "mod_dips"]
    out += ["voice_ratio", "voice_f0_median", "voice_f0_std_st", "voice_regions", "voice_hnr_mean"]
    out += [f"voice_hnr_s{s}" for s in range(N_SEG)]
    out += ["dur_s"]
    return out


def features(y):
    y = y.astype(np.float64)
    y = y - y.mean()
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.05  # RMS normalisation
    if len(y) < N_FFT + HOP:
        y = np.pad(y, (0, N_FFT + HOP - len(y)))

    # ---- spectrogram
    n_frames = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = y[idx]
    P = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1)) ** 2 + 1e-12
    logmel = 10 * np.log10(P @ mel_fb().T + 1e-12)
    logmel -= logmel.mean()  # remove overall level only
    f_grid = seg_means(logmel).ravel()

    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    d1 = np.gradient(mfcc, axis=0) if n_frames > 1 else np.zeros_like(mfcc)
    d2 = np.gradient(d1, axis=0) if n_frames > 1 else np.zeros_like(mfcc)
    f_mfcc = np.concatenate([np.concatenate([m.mean(0), m.std(0)]) for m in (mfcc, d1, d2)])

    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    tot = P.sum(1)
    cent = (P * freqs).sum(1) / tot
    spread = np.sqrt((P * (freqs[None, :] - cent[:, None]) ** 2).sum(1) / tot)
    roll = freqs[np.argmax(np.cumsum(P, 1) >= 0.85 * tot[:, None], axis=1)]
    flat = np.exp(np.log(P).mean(1)) / P.mean(1)
    zcr = (np.abs(np.diff(np.sign(frames), axis=1)) > 0).mean(1)
    en = 10 * np.log10(tot)
    en -= en.max()
    desc = np.stack([cent, spread, roll, flat, zcr, en], axis=1)
    f_spec = np.concatenate([np.stack([desc.mean(0), desc.std(0)], axis=1).ravel(),
                             seg_means(desc[:, [0, 3, 4, 5]]).T.ravel()])

    # ---- amplitude modulation (trill)
    band = sosfiltfilt(butter(4, [300, 3000], btype="band", fs=SR, output="sos"), y)
    env = np.abs(hilbert(band))
    env = sosfiltfilt(butter(4, 60, btype="low", fs=SR, output="sos"), env)[:: SR // ENV_SR]
    env = np.maximum(env, 0)
    m_env = env.mean() + 1e-12
    spec = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(len(env)), 256)) ** 2
    mf = np.fft.rfftfreq(256, 1 / ENV_SR)
    total = spec[(mf >= 2) & (mf <= 60)].sum() + 1e-12
    fr = [spec[(mf >= a) & (mf < b)].sum() / total for a, b in ((2, 8), (8, 15), (15, 35), (35, 60))]
    win, hop = 24, 4  # 120 ms window, 20 ms hop
    trill = []
    wf = np.fft.rfftfreq(64, 1 / ENV_SR)
    for s in range(0, max(1, len(env) - win + 1), hop):
        w = env[s:s + win]
        if len(w) < win:
            break
        sp = np.abs(np.fft.rfft((w - w.mean()) * np.hanning(win), 64))
        trill.append(sp[(wf >= 15) & (wf <= 35)].max() / (w.mean() * win / 2 + 1e-12))
    trill = np.array(trill) if trill else np.zeros(1)
    try:
        rip = sosfiltfilt(butter(2, [15, 35], btype="band", fs=ENV_SR, output="sos"), env)
        peaks, _ = find_peaks(rip, prominence=0.15 * m_env)
        n_peaks = len(peaks)
    except ValueError:
        n_peaks = 0
    env_db = 20 * np.log10(env / m_env + 1e-6)
    dips, _ = find_peaks(-env_db, prominence=6, width=(None, 0.04 * ENV_SR))
    dur = len(y) / SR
    f_mod = np.array(fr + [trill.max(), trill.argmax() / max(1, len(trill) - 1), trill.mean(),
                           n_peaks, n_peaks / dur, len(dips)])

    # ---- voice (Praat)
    import parselmouth
    snd = parselmouth.Sound(y, sampling_frequency=SR)
    f0 = snd.to_pitch(time_step=0.01, pitch_floor=100, pitch_ceiling=600).selected_array["frequency"]
    voiced = f0 > 0
    regions = int(np.sum(np.diff(np.concatenate([[0], voiced.astype(int)])) == 1))
    if voiced.any():
        f0_med = float(np.median(f0[voiced]))
        f0_std = float(np.std(12 * np.log2(f0[voiced] / f0_med)))
    else:
        f0_med, f0_std = 0.0, 0.0
    hnr = np.clip(snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=100).values[0], -10, 40)
    if len(hnr) == 0:
        hnr = np.zeros(1)
    f_voice = np.concatenate([[voiced.mean() if len(voiced) else 0.0, f0_med, f0_std, regions, hnr.mean()],
                              seg_means(hnr[:, None]).ravel()])

    return np.concatenate([f_grid, f_mfcc, f_spec, f_mod, f_voice, [dur]]).astype(np.float32)


def work(path):
    try:
        y, sr = sf.read(CHILD / path, dtype="float64")
        assert sr == SR
        return features(y)
    except Exception as e:  # noqa: BLE001
        return f"{path}: {type(e).__name__} {e}"


def keep_awake():
    """Ask Windows not to idle-sleep while this script runs. Ends when the process exits."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="dsp_v1.npz")
    ap.add_argument("--sets", default="train,test_voices,confidence", help="sets from data/splits.csv")
    a = ap.parse_args()
    keep_awake()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(ROOT / "data" / "splits.csv", encoding="utf-8") as f:
        keep = {r["clip_id"] for r in csv.DictReader(f) if r["set"] in a.sets.split(",")}
    with open(CHILD / "metadata.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["clip_id"] in keep]
    if a.limit:
        rows = rows[:: max(1, len(rows) // a.limit)][: a.limit]

    feat_names = names()
    X = np.zeros((len(rows), len(feat_names)), dtype=np.float32)
    ok = np.ones(len(rows), dtype=bool)
    errors, t0 = [], time.time()
    print(f"{len(rows)} clips, {len(feat_names)} features", flush=True)
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(work, [r["path"] for r in rows], chunksize=32)):
            if isinstance(r, str):
                errors.append(r)
                ok[i] = False
            else:
                X[i] = r
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(rows)}  {(i + 1) / (time.time() - t0):.0f} clips/s", flush=True)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    ids = np.array([r["clip_id"] for r in rows])
    np.savez_compressed(OUT / a.out, X=X[ok], clip_ids=ids[ok], names=np.array(feat_names))
    print(f"done: {int(ok.sum())} clips x {X.shape[1]} features in {time.time() - t0:.0f}s, "
          f"{len(errors)} failed -> {OUT / a.out}")
    for e in errors[:10]:
        print("  FAIL", e)


if __name__ == "__main__":
    main()
