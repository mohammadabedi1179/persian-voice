"""Transcribe the real call recordings (no reference text exists, so nothing is scored).

    python -m scripts.transcribe_real samples-real samples-real-results

Writes one <call>.txt per call (mixed, in and out tracks) and summary.tsv. Calls already
written are skipped, so an interrupted run can be restarted. Prints no transcript text.
"""
import csv
import sys
from pathlib import Path

from pv.stt import Transcriber

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
calls = sorted({p.stem.removesuffix("-in").removesuffix("-out") for p in src.glob("*.wav")})
tracks = [("mixed", ""), ("in", "-in"), ("out", "-out")]
tr = Transcriber("fa-turbo")

for n, call in enumerate(calls, 1):
    dst = out / f"{call}.txt"
    if dst.exists():
        continue
    parts, rows = [], []
    for name, suffix in tracks:
        wav = src / f"{call}{suffix}.wav"
        if not wav.exists():
            continue
        r = tr.transcribe(str(wav))
        parts.append(f"=== {name}: {wav.name} ({r['audio_seconds']} s) ===\n{r['text']}\n")
        rows.append({"call": call, "track": name, "audio_seconds": r["audio_seconds"],
                     "seconds": r["seconds"], "rtf": r["rtf"], "words": len(r["text"].split())})
    dst.write_text("\n".join(parts), encoding="utf-8")
    summary = out / "summary.tsv"
    new = not summary.exists()
    with open(summary, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys(), delimiter="\t")
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"{n}/{len(calls)} {call} {sum(r['audio_seconds'] for r in rows):.0f} s audio "
          f"in {sum(r['seconds'] for r in rows):.0f} s", flush=True)
print("REAL_DONE", flush=True)
