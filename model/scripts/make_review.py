"""
Build a listening page to check generated clips by ear.

  python scripts/make_review.py              -> data/child/review.html (child-like clips)
  python scripts/make_review.py --set raw    -> data/raw/review.html   (adult TTS clips)

One row per (word, voice). The correct clip (ر) comes first, then every error variant.
With many clips, only a random sample of rows is shown (--rows). Tick "مشكلة" on bad
clips and press the export button: it downloads flagged_clips.txt. Put that file in
data/ (it is merged into data/excluded_clips.txt by the training step).

  python scripts/make_review.py --rows 150 --only unverified   (foreign voices' correct clips)
  python scripts/make_review.py --only distortion               (in-between R sounds)
"""

import argparse
import base64
import csv
import html
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["child", "raw", "blend"], default="child")
    ap.add_argument("--rows", type=int, default=150, help="max rows shown (random sample)")
    ap.add_argument("--only", choices=["all", "unverified", "distortion"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-embed", action="store_true",
                    help="link the WAV files instead of embedding them (page is smaller, but some browsers block local audio)")
    ap.add_argument("--name", default="review.html", help="output file name")
    a = ap.parse_args()
    base = ROOT / "data" / a.set
    voice_col = "child_voice" if a.set == "child" else "voice"

    with open(base / "metadata.csv", encoding="utf-8") as f:
        meta = [m for m in csv.DictReader(f) if m.get("utt_id")]
    # leave out voices and clips that are already excluded from training
    def read_list(name):
        p = ROOT / "data" / name
        if not p.exists():
            return set()
        return {l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
    bad_voices, bad_clips = read_list("excluded_voices.txt"), read_list("excluded_clips.txt")
    meta = [m for m in meta if m["voice_group"] not in bad_voices and m["clip_id"] not in bad_clips]
    if a.only == "unverified":
        meta = [m for m in meta if m.get("review_status") == "unverified"]
    elif a.only == "distortion":
        keep = {(m["word_id"], m[voice_col]) for m in meta if m["tier"] == "distortion"}
        meta = [m for m in meta if (m["word_id"], m[voice_col]) in keep
                and m["tier"] in ("distortion", "vocoded")]

    groups = defaultdict(list)
    for m in meta:
        groups[(m["word_id"], m[voice_col])].append(m)

    # sample evenly per word, so every word is on the page; voices rotate across words
    keys = sorted(groups)
    if len(keys) > a.rows:
        rng = random.Random(a.seed)
        by_word = defaultdict(list)
        for k in keys:
            by_word[k[0]].append(k)
        per_word = max(1, -(-a.rows // len(by_word)))
        keys = []
        for i, (wid, ks) in enumerate(sorted(by_word.items())):
            ks = sorted(ks, key=lambda k: k[1])
            start = (i * per_word) % len(ks)  # rotate so different words show different voices
            rot = ks[start:] + ks[:start]
            keys += rot[:per_word] if len(rot) <= per_word else sorted(rng.sample(rot[:per_word * 2], per_word))
        keys = sorted(keys)
    shown = sum(len(groups[k]) for k in keys)
    def src(path):
        if a.no_embed:
            return html.escape(path)
        return "data:audio/wav;base64," + base64.b64encode((base / path).read_bytes()).decode()

    rows = []
    for (wid, voice), items in ((k, groups[k]) for k in keys):
        items.sort(key=lambda m: (m["label"] != "correct", m["tier"] != "common", m["produced"]))
        ref = items[0]
        params = ""
        if a.set == "child":
            params = (f"<div class='s'>formants ×{ref['formant_shift']} · F0 {float(ref['pitch_median']):.0f} Hz"
                      f" · speed ×{ref['duration']}</div>")
        cells = []
        for m in items:
            cls = {"correct": "ok", "distorted": "mid"}.get(m["label"], "err")
            if m["label"] == "correct":
                tag = "صحيح" + (" (يحتاج تأكيد)" if m.get("review_status") == "unverified" else "")
            elif m["label"] == "distorted":
                tag = f"بين بين: {html.escape(m['produced_char'])} ({float(m['blend_alpha']):.0%})"
            else:
                tag = f"بدل الراء: {html.escape(m['produced_char'])}"
            if m.get("tier") == "vocoded":
                tag += " · vocoder"
            cells.append(
                f"<div class='clip {cls}'><div class='t'>{html.escape(m['text'])}</div>"
                f"<div class='s'>{tag}</div>"
                f"<audio controls preload='none' src='{src(m['path'])}'></audio>"
                f"<label class='s'><input type='checkbox' data-clip='{html.escape(m['clip_id'])}'> مشكلة</label></div>")
        rows.append(
            f"<tr><td class='w'>{wid}<div class='t'>{html.escape(ref['word_plain'])}</div>"
            f"<div class='s'>{ref['position']}</div></td>"
            f"<td class='v'>{html.escape(voice)}{params}</td>"
            f"<td><div class='clips'>{''.join(cells)}</div></td></tr>")

    title = {"child": "أصوات الأطفال (مولّدة)", "raw": "أصوات البالغين (TTS)", "blend": "راء بين بين (بالغين)"}[a.set]
    page = f"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>مراجعة الأصوات</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#666; --line:#ddd; --head:#eef3f8; --ok:#e7f4ea; --err:#fbeceb; --mid:#fdf5dc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#eee; --muted:#aaa; --line:#333; --head:#1f2a35; --ok:#1d2e22; --err:#33201f; --mid:#332c17; }} }}
body {{ font-family: Arial, sans-serif; background:var(--bg); color:var(--fg); margin:16px; }}
table {{ border-collapse: collapse; width:100%; }}
th, td {{ border:1px solid var(--line); padding:6px 8px; vertical-align:top; }}
th {{ background:var(--head); position:sticky; top:0; z-index:1; }}
.w {{ width:80px; }} .v {{ width:190px; font-size:13px; word-break:break-all; }}
.t {{ font-size:19px; }} .s {{ color:var(--muted); font-size:12px; }}
.clips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.clip {{ border:1px solid var(--line); border-radius:6px; padding:4px 6px; width:170px; }}
.clip.ok {{ background:var(--ok); }} .clip.err {{ background:var(--err); }} .clip.mid {{ background:var(--mid); }}
.bar {{ position:sticky; top:0; background:var(--bg); padding:8px 0; z-index:2; display:flex; gap:12px; align-items:center; }}
button {{ font:inherit; padding:6px 14px; cursor:pointer; }}
audio {{ width:160px; height:30px; }}
</style></head><body>
<h1>مراجعة: {title}</h1>
<p>الأخضر النطق الصحيح، والأحمر نطقات فيها الراء متبدلة أو محذوفة. تأكد أن كل مقطع قال المكتوب فوقه، وأن الصوت يشبه طفل.</p>
<div class="bar"><button id="exp">تصدير المقاطع المعلّمة</button><span id="cnt" class="s">0 معلّم</span>
<span class="s">معروض {len(keys)} صف من {len(groups)} ({shown} مقطع من {len(meta)})</span></div>
<table><thead><tr><th>الكلمة</th><th>الصوت</th><th>المقاطع</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<script>
const KEY = "nutq-flags-{a.set}";
let saved = [];
try {{ saved = JSON.parse(localStorage.getItem(KEY) || "[]"); }} catch (e) {{}}
const boxes = [...document.querySelectorAll("input[data-clip]")];
const cnt = document.getElementById("cnt");
function update() {{
  const on = boxes.filter(b => b.checked).map(b => b.dataset.clip);
  cnt.textContent = on.length + " معلّم";
  try {{ localStorage.setItem(KEY, JSON.stringify(on)); }} catch (e) {{}}
  return on;
}}
boxes.forEach(b => {{ b.checked = saved.includes(b.dataset.clip); b.addEventListener("change", update); }});
update();
document.getElementById("exp").addEventListener("click", () => {{
  const blob = new Blob([update().join("
") + "
"], {{type: "text/plain"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "flagged_clips.txt"; a.click();
}});
</script></body></html>"""
    out = base / a.name
    out.write_text(page, encoding="utf-8")
    print("wrote", out, f"{len(keys)} of {len(groups)} rows, {shown} clips shown,",
          f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
