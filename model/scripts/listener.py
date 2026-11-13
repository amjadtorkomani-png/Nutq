"""
سامع النطق: the listener used by the app and by predict.py.

One recording + the word the child was asked to say -> صح / قريب / غلط

  1. audio      any sample rate -> 16 kHz mono -> silence trimmed (50 ms kept)
                -> optional child voice (Praat "Change gender") -> peak-limited
  2. embedding  mHuBERT-147: mean of layers 6, 9, 12 + layer 9 in 3 time segments = 4608 numbers
  3. 3 models   on the same 4608 numbers (all logistic regression):
                  r_model   P(the R is wrong), calibrated (isotonic)
                  word      which of the 25 words was said
                  letter    which sound is in the R place (r, ل, غ, ... or none)
  4. decision   غلط  if the word is not the asked word
                غلط  if the letter model hears another sound than R
                otherwise by P(R wrong):  < 0.1 صح   |  0.1 - 0.9 قريب (to the specialist)  |  > 0.9 غلط

Model file: models/nutq_r.joblib (made by train_model.py)
"""

import io
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".hf_cache").exists():
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

SR = 16000
MODEL_FILE = next((p for p in (ROOT / "models" / "nutq_r.joblib", ROOT / "trained" / "nutq_r.joblib") if p.exists()),
                  ROOT / "models" / "nutq_r.joblib")
EMBEDDING_MODEL = "utter-project/mHuBERT-147"
MEAN_LAYERS = (6, 9, 12)
SEG_LAYER, N_SEG = 9, 3
CHILD = dict(formant_shift=1.2, pitch_median=275.0, pitch_range=1.0, duration=1.0)


# ---------------------------------------------------------------- audio
def trim(audio, sr, thresh_db=-35.0, pad_s=0.05):
    fr = int(0.02 * sr)
    if len(audio) < fr:
        return audio
    frames = len(audio) // fr
    e = np.array([np.sqrt(np.mean(audio[i * fr:(i + 1) * fr] ** 2)) for i in range(frames)])
    e_db = 20 * np.log10(e + 1e-9)
    on = np.where(e_db > e_db.max() + thresh_db)[0]
    if len(on) == 0:
        return audio
    pad = int(pad_s * sr)
    return audio[max(0, on[0] * fr - pad):min(len(audio), (on[-1] + 1) * fr + pad)]


def resample(y, sr_in, sr_out=SR):
    if int(sr_in) == sr_out:
        return y
    from scipy.signal import resample_poly
    g = np.gcd(int(sr_in), int(sr_out))
    return resample_poly(y, sr_out // g, int(sr_in) // g)


def to_child(y, sr=SR):
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    out = call(snd, "Change gender", 75, 600, CHILD["formant_shift"], CHILD["pitch_median"],
               CHILD["pitch_range"], CHILD["duration"])
    return resample(out.values[0], out.sampling_frequency)


def load_audio(src):
    """src: path or bytes (wav / flac / ogg / mp3)"""
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return resample(y - y.mean(), sr).astype(np.float32)


# ---------------------------------------------------------------- embedding
def embed(model, torch, y):
    """4608 numbers per clip (same function is used to make the training features)"""
    y = y.astype(np.float32)
    y = y - y.mean()
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.05  # same loudness for every clip
    if len(y) < 4000:
        y = np.pad(y, (0, 4000 - len(y)))
    with torch.inference_mode():
        hs = model(torch.from_numpy(y)[None], output_hidden_states=True).hidden_states
    parts = [hs[l][0].mean(0).numpy() for l in MEAN_LAYERS]
    h = hs[SEG_LAYER][0].numpy()
    parts += [h[idx].mean(0) for idx in np.array_split(np.arange(len(h)), N_SEG)]
    return np.concatenate(parts).astype(np.float16)


# ---------------------------------------------------------------- decision
def decide(b, p_r, p_word, p_letter, target_word):
    """b = model bundle; p_word / p_letter = probability rows of the word / letter model"""
    t = b["thresholds"]
    words, letters = list(b["word_model"].classes_), list(b["letter_model"].classes_)
    heard = words[int(np.argmax(p_word))]
    p_target = float(p_word[words.index(target_word)]) if target_word in words else None
    top_letter = letters[int(np.argmax(p_letter))]
    reason = ""
    if target_word and heard != target_word and (p_target or 0) < t["word_min"]:
        verdict, reason = "incorrect", "word"
    elif top_letter != "r" and float(np.max(p_letter)) >= t["letter_min"]:
        verdict, reason = "incorrect", "letter"
    elif p_r < t["lo"]:
        verdict = "correct"
    elif p_r > t["hi"]:
        verdict, reason = "incorrect", "r"
    else:
        verdict = "close"
    return {"verdict": verdict, "reason": reason, "p_error": float(p_r), "heard_word": heard,
            "p_target": p_target, "letter": top_letter, "p_letter": float(np.max(p_letter))}


class Listener:
    def __init__(self, model_file=MODEL_FILE, threads=4):
        import joblib
        import torch
        from transformers import HubertModel
        torch.set_num_threads(threads)
        self.torch = torch
        self.hubert = HubertModel.from_pretrained(EMBEDDING_MODEL).eval()
        self.b = joblib.load(model_file)

    def judge(self, src, target_word="", adult=True):
        y = load_audio(src)
        warn = []
        if np.sqrt(np.mean(y ** 2)) < 0.003:
            warn.append("quiet")
        y = trim(y, SR)
        dur = len(y) / SR
        if dur < 0.25:
            warn.append("short")
        if dur > 2.0:
            warn.append("long")
        if adult:
            y = to_child(y)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        if peak > 0.95:
            y = y * (0.95 / peak)
        e = embed(self.hubert, self.torch, y).astype(np.float32)[None]
        b = self.b
        p_r = float(b["calibrator"].predict(b["r_model"].predict_proba(e)[:, 1])[0])
        out = decide(b, p_r, b["word_model"].predict_proba(e)[0], b["letter_model"].predict_proba(e)[0], target_word)
        out.update(duration_s=round(dur, 3), warnings=warn)
        return out
