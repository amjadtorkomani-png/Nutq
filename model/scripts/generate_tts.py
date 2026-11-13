"""
Generate synthetic (adult-voice) clips for the R task (سامع النطق).

Reads data/utterances_r.csv (made by build_utterances.py). For every word and
every voice it makes:
  - the correct utterance (ر)
  - N error utterances sampled per (voice, word); common errors are picked
    twice as often as other letters. Use --all-errors to make every variant.

Engines (each one is skipped if it is not usable):
  edge        edge-tts, no key (unofficial use of the Edge read-aloud service)
  azure       Azure Speech, needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
  elevenlabs  ElevenLabs, needs ELEVENLABS_API_KEY

Output: 16 kHz mono 16-bit WAV in data/raw/<engine>/<voice>/<utt_id>.wav and
data/raw/metadata.csv. These are ADULT voices: run childify.py next.

Split train/validation by `voice_group`. Azure and edge share the same
Microsoft voices, so a voice gets the same group in both engines.

Usage:
  python scripts/generate_tts.py --engines edge --max-words 4 --max-voices 3 --all-errors
  python scripts/generate_tts.py --engines all --errors-per-word 4
"""

import argparse
import asyncio
import csv
import hashlib
import io
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
ROOT = Path(__file__).resolve().parents[1]

META_FIELDS = [
    "clip_id", "engine", "voice", "voice_group", "locale", "utt_id", "word_id", "position",
    "word_plain", "label", "produced", "produced_char", "tier", "text", "path", "sr",
    "duration_s", "peak", "rms_db", "qc_flag", "review_status",
]


# ---------------------------------------------------------------- helpers
def load_utterances(path, max_words=None, word_ids=None):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_word = defaultdict(list)
    for r in rows:
        by_word[r["word_id"]].append(r)
    ids = [w for w in sorted(by_word) if not word_ids or w in word_ids][: max_words or None]
    return {w: by_word[w] for w in ids}


def pick(utts, voice, n_errors, all_errors):
    """correct + sampled errors, deterministic per (voice, word)"""
    correct = [u for u in utts if u["label"] == "correct"]
    errors = [u for u in utts if u["label"] == "incorrect"]
    if all_errors or n_errors >= len(errors):
        return correct + errors
    seed = int(hashlib.md5(f"{voice}|{utts[0]['word_id']}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    pool = list(errors)
    chosen = []
    for _ in range(n_errors):
        w = [2.0 if u["tier"] == "common" else 1.0 for u in pool]
        u = rng.choices(pool, weights=w, k=1)[0]
        chosen.append(u)
        pool.remove(u)
    return correct + chosen


def to_target(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return audio


def qc(audio):
    dur = len(audio) / TARGET_SR
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    rms_db = 20 * np.log10(rms + 1e-9)
    flags = []
    if dur < 0.25:
        flags.append("too_short")
    if dur > 3.5:
        flags.append("too_long")
    if rms_db < -45:
        flags.append("too_quiet")
    if peak >= 0.999:
        flags.append("clipped")
    return dur, peak, rms_db, ";".join(flags)


class Writer:
    def __init__(self, out_dir, overwrite):
        self.out = out_dir
        self.overwrite = overwrite
        self.meta_path = out_dir / "metadata.csv"
        self.meta = {}
        if self.meta_path.exists():
            with open(self.meta_path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if "utt_id" in r and r["utt_id"]:  # ignore old-schema rows
                        self.meta[r["clip_id"]] = r
        self.made = 0
        self.skipped = 0
        self.failed = []

    def clip_path(self, engine, voice, utt_id):
        return self.out / engine / voice / f"{utt_id}.wav"

    def exists(self, engine, voice, utt_id):
        return (not self.overwrite) and self.clip_path(engine, voice, utt_id).exists()

    def adopt(self, engine, voice, voice_group, locale, utt, review_status="ok"):
        """file already on disk (e.g. from an interrupted run): make sure it has a metadata row"""
        if f"{engine}__{voice}__{utt['utt_id']}" in self.meta:
            return
        audio, sr = sf.read(self.clip_path(engine, voice, utt["utt_id"]), dtype="float32")
        self.save(engine, voice, voice_group, locale, utt, audio, sr, review_status)
        self.made -= 1

    def save(self, engine, voice, voice_group, locale, utt, audio, sr, review_status="ok"):
        audio = to_target(audio, sr)
        p = self.clip_path(engine, voice, utt["utt_id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, audio, TARGET_SR, subtype="PCM_16")
        dur, peak, rms_db, flag = qc(audio)
        clip_id = f"{engine}__{voice}__{utt['utt_id']}"
        self.meta[clip_id] = {
            "clip_id": clip_id, "engine": engine, "voice": voice, "voice_group": voice_group,
            "locale": locale, **{k: utt[k] for k in ("utt_id", "word_id", "position", "word_plain",
                                                    "label", "produced", "produced_char", "tier", "text")},
            "path": p.relative_to(self.out).as_posix(), "sr": TARGET_SR,
            "duration_s": f"{dur:.3f}", "peak": f"{peak:.3f}", "rms_db": f"{rms_db:.1f}",
            "qc_flag": flag, "review_status": review_status,
        }
        self.made += 1

    def write_meta(self):
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=META_FIELDS)
            w.writeheader()
            for k in sorted(self.meta):
                w.writerow(self.meta[k])


# ---------------------------------------------------------------- engines
async def _edge_all(words, writer, a):
    import edge_tts

    allv = await edge_tts.list_voices()
    voices = sorted([v for v in allv if v["Locale"].startswith("ar-")], key=lambda v: v["ShortName"])
    voices = voices[: a.max_voices or None]
    if a.multilingual:
        ml = sorted([v for v in allv if "Multilingual" in v["ShortName"]], key=lambda v: v["ShortName"])
        voices += ml[: a.max_voices or None]
    rates = a.rates.split(",")
    sem = asyncio.Semaphore(a.concurrency)

    async def one(v, rate, utt):
        base = v["ShortName"]
        name = base if rate in ("+0%", "0%") else f"{base}_rate{rate.replace('%', '')}"
        is_ml = "Multilingual" in base
        # an Arabic R from a foreign voice is not guaranteed -> needs a human check
        status = "unverified" if (is_ml and utt["label"] == "correct") else "ok"
        if writer.exists("edge", name, utt["utt_id"]):
            writer.adopt("edge", name, base, v["Locale"], utt, status)
            writer.skipped += 1
            return
        async with sem:
            for attempt in range(3):
                try:
                    buf = bytearray()

                    async def fetch():
                        async for chunk in edge_tts.Communicate(utt["text"], base, rate=rate).stream():
                            if chunk["type"] == "audio":
                                buf.extend(chunk["data"])

                    await asyncio.wait_for(fetch(), timeout=20)  # a stuck request must not block a slot
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as t:
                        t.write(buf)
                        tmp = t.name
                    try:
                        audio, sr = sf.read(tmp, dtype="float32")
                    finally:
                        os.unlink(tmp)
                    writer.save("edge", name, base, v["Locale"], utt, audio, sr, status)
                    return
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        writer.failed.append(f"edge/{name}/{utt['utt_id']}: {type(e).__name__} {e}")
                    await asyncio.sleep(1.5 * (attempt + 1))

    jobs = [one(v, rate, u) for v in voices for rate in rates for w in words.values()
            for u in pick(w, f"{v['ShortName']}|{rate}", a.errors_per_word, a.all_errors)]
    done = 0

    async def tracked(j):
        nonlocal done
        await j
        done += 1
        if done % 500 == 0:
            print(f"  edge {done}/{len(jobs)}", flush=True)
            writer.write_meta()

    await asyncio.gather(*(tracked(j) for j in jobs))
    print(f"  edge: {len(voices)} voices x {len(rates)} rates, {len(jobs)} clips requested")


def run_edge(words, writer, a):
    asyncio.run(_edge_all(words, writer, a))


def run_azure(words, writer, a):
    key, region = os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        print("  azure skipped: set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION")
        return
    import azure.cognitiveservices.speech as speechsdk

    cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    cfg.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
    synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
    vres = synth.get_voices_async("").get()
    voices = sorted([v for v in vres.voices if v.locale.startswith("ar-")], key=lambda v: v.short_name)
    for v in voices[: a.max_voices or None]:
        for w in words.values():
            for utt in pick(w, v.short_name, a.errors_per_word, a.all_errors):
                if writer.exists("azure", v.short_name, utt["utt_id"]):
                    writer.skipped += 1
                    continue
                ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                        f'xml:lang="{v.locale}"><voice name="{v.short_name}">{utt["text"]}</voice></speak>')
                r = synth.speak_ssml_async(ssml).get()
                if r.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                    writer.failed.append(f"azure/{v.short_name}/{utt['utt_id']}: {r.reason}")
                    continue
                audio, sr = sf.read(io.BytesIO(r.audio_data), dtype="float32")
                writer.save("azure", v.short_name, v.short_name, v.locale, utt, audio, sr)
        print(f"  azure {v.short_name} done")


def run_elevenlabs(words, writer, a):
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("  elevenlabs skipped: set ELEVENLABS_API_KEY")
        return
    import requests

    h = {"xi-api-key": key}
    r = requests.get("https://api.elevenlabs.io/v1/voices", headers=h, timeout=30)
    r.raise_for_status()
    voices = sorted(r.json()["voices"], key=lambda v: v["name"])[: a.max_voices or 10]
    for v in voices:
        vname = "el-" + "".join(c if c.isalnum() else "-" for c in v["name"]).strip("-")
        for w in words.values():
            for utt in pick(w, vname, a.errors_per_word, a.all_errors):
                if writer.exists("elevenlabs", vname, utt["utt_id"]):
                    writer.skipped += 1
                    continue
                resp = requests.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{v['voice_id']}",
                    params={"output_format": "pcm_16000"}, headers=h, timeout=60,
                    json={"text": utt["text"], "model_id": "eleven_multilingual_v2"})
                if resp.status_code != 200:
                    writer.failed.append(f"elevenlabs/{vname}/{utt['utt_id']}: "
                                         f"{resp.status_code} {resp.text[:120]}")
                    continue
                audio = np.frombuffer(resp.content, dtype="<i2").astype(np.float32) / 32768.0
                writer.save("elevenlabs", vname, vname, "multi", utt, audio, 16000)
        print(f"  elevenlabs {vname} done")


ENGINES = {"edge": run_edge, "azure": run_azure, "elevenlabs": run_elevenlabs}


def keep_awake():
    """Ask Windows not to idle-sleep while this script runs (like a video player does).
    No setting is changed; the request ends when the process exits."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances", default=str(ROOT / "data" / "utterances_r.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--engines", default="all", help="comma list or 'all'")
    ap.add_argument("--max-words", type=int, default=None)
    ap.add_argument("--word-ids", default=None, help="comma list, e.g. w01,w12,w19")
    ap.add_argument("--max-voices", type=int, default=None, help="per engine")
    ap.add_argument("--errors-per-word", type=int, default=4)
    ap.add_argument("--all-errors", action="store_true")
    ap.add_argument("--multilingual", action="store_true", help="add edge Multilingual voices")
    ap.add_argument("--concurrency", type=int, default=12, help="parallel edge requests")
    ap.add_argument("--rates", default="+0%", help="edge speaking rates, e.g. +0%%,-20%%")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    keep_awake()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    words = load_utterances(a.utterances, a.max_words,
                            a.word_ids.split(",") if a.word_ids else None)
    engines = list(ENGINES) if a.engines == "all" else a.engines.split(",")
    writer = Writer(Path(a.out), a.overwrite)

    for e in engines:
        print(f"[{e}]")
        try:
            ENGINES[e](words, writer, a)
        except Exception as ex:  # noqa: BLE001
            writer.failed.append(f"{e}: engine error: {ex}")
            print(f"  {e} failed: {ex}")
        writer.write_meta()

    flagged = [m for m in writer.meta.values() if m["qc_flag"]]
    print(f"\nmade {writer.made}, skipped (already there) {writer.skipped}, "
          f"failed {len(writer.failed)}, total in metadata {len(writer.meta)}, qc-flagged {len(flagged)}")
    for f in writer.failed[:15]:
        print("  FAIL", f)
    for m in flagged[:15]:
        print("  QC  ", m["clip_id"], m["qc_flag"])


if __name__ == "__main__":
    main()
