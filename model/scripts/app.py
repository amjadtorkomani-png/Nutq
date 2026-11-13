"""
سامع النطق: local web app to try the R model with your own voice.

The page shows a word to say (correctly, or with a given error), records you, and the listener answers
صح / قريب / غلط (see listener.py for how). It also shows whether that matches what you were asked to say.

Every recording is kept in data/real/ (wav + one row in data/real/log.csv) for testing the model on
real voices.

Usage: python scripts/app.py        then open http://127.0.0.1:8765 in Chrome or Edge
"""

import csv
import json
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import listener  # noqa: E402

ROOT = listener.ROOT
PAGE = ROOT / "app" / "index.html"
REAL = ROOT / "data" / "real"
PORT = 8765
LOG_COLS = ["time", "file", "mode", "utt_id", "word_id", "expected", "verdict", "p_error", "p_raw", "adult",
            "duration_s", "model", "heard_word", "p_target", "letter", "p_letter", "reason"]


def utterances():
    words = {r["word_id"]: r for r in csv.DictReader(open(ROOT / "data" / "words_r.csv", encoding="utf-8"))}
    out = {}
    for r in csv.DictReader(open(ROOT / "data" / "utterances_r.csv", encoding="utf-8")):
        w = words[r["word_id"]]
        text = r["text"]
        if w["word_plain"].endswith("ة") and text.endswith("ه"):
            text = text[:-1] + "ة"  # the TTS got ه, a person reads ة
        out.setdefault(r["word_id"], {"word_id": r["word_id"], "word": w["word_plain"], "correct_text": w["correct_text"],
                                      "variants": []})
        out[r["word_id"]]["variants"].append({"utt_id": r["utt_id"], "label": r["label"], "text": text,
                                              "produced": r["produced"], "produced_char": r["produced_char"]})
    return list(out.values())


def log_row(row):
    REAL.mkdir(parents=True, exist_ok=True)
    f = REAL / "log.csv"
    if f.exists():  # older log with other columns -> rewrite once with the new header, rows kept
        with open(f, encoding="utf-8") as fh:
            head = fh.readline().strip().split(",")
        if head != LOG_COLS:
            with open(f, encoding="utf-8") as fh:
                old = list(csv.reader(fh))[1:]
            with open(f, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(LOG_COLS)
                for r in old:
                    w.writerow((r + [""] * len(LOG_COLS))[:len(LOG_COLS)])
    new = not f.exists()
    with open(f, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/utterances":
            return self._send(200, UTTS)
        if path == "/api/health":
            return self._send(200, {"ok": True, "model": listener.MODEL_FILE.name})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/judge":
            return self._send(404, {"error": "not found"})
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            t0 = time.time()
            with LOCK:
                res = LISTENER.judge(data, target_word=q.get("word_id", ""), adult=q.get("adult", "1") == "1")
            res["ms"] = int((time.time() - t0) * 1000)
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"ما قدرت أقرأ التسجيل: {e}"})
        if q.get("save") == "0":  # re-checking an existing recording: nothing is written
            return self._send(200, res)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ctype = self.headers.get("Content-Type", "")
        ext = next((x for k, x in (("mpeg", ".mp3"), ("mp3", ".mp3"), ("ogg", ".ogg"), ("flac", ".flac")) if k in ctype), ".wav")
        name = f"{stamp}_{q.get('utt_id', 'free')}{ext}"
        REAL.mkdir(parents=True, exist_ok=True)
        (REAL / name).write_bytes(data)  # the original recording, before any processing
        log_row({"time": stamp, "file": name, "mode": q.get("mode", ""), "utt_id": q.get("utt_id", ""),
                 "word_id": q.get("word_id", ""), "expected": q.get("expected", ""), "verdict": res["verdict"],
                 "p_error": f"{res['p_error']:.4f}", "adult": q.get("adult", "1"), "duration_s": res["duration_s"],
                 "model": listener.MODEL_FILE.name, "heard_word": res["heard_word"],
                 "p_target": "" if res["p_target"] is None else f"{res['p_target']:.4f}",
                 "letter": res["letter"], "p_letter": f"{res['p_letter']:.4f}", "reason": res["reason"]})
        res["file"] = name
        self._send(200, res)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("loading model ...", flush=True)
    LISTENER = listener.Listener()
    LOCK = threading.Lock()
    UTTS = utterances()
    print(f"ready: http://127.0.0.1:{PORT}  ({len(UTTS)} words)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
