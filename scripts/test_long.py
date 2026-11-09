"""Transcribe one long file built from FLEURS sentences and compare with sentence-by-sentence scoring.

    python -m scripts.test_long --minutes 10
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from pv.audio import load_audio, save_wav
from pv.metrics import wer_cer
from pv.stt import SR, Transcriber, split_on_pauses

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=10)
minutes = ap.parse_args().minutes

with open("samples/fleurs_test.tsv", encoding="utf-8") as f:
    paths = [r[0] for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
with open("results/fleurs.fa-turbo.tsv", encoding="utf-8") as f:
    scored = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

gap = np.zeros(int(0.4 * SR), np.float32)
wavs, refs, hyps, total = [], [], [], 0.0
for path, (_, ref, hyp) in zip(paths, scored):
    if total >= minutes * 60:
        break
    w = load_audio(path, SR)
    wavs += [w, gap]
    refs.append(ref)
    hyps.append(hyp)
    total += len(w) / SR + 0.4
long_wav = np.concatenate(wavs[:-1])
out = save_wav(Path("results") / f"long_{minutes:g}min.wav", long_wav, SR)
segments = split_on_pauses(long_wav)
print(f"{len(refs)} sentences, {total / 60:.1f} min, {len(segments)} segments, "
      f"longest {max(len(s) for s in segments) / SR:.1f} s", flush=True)

r = Transcriber("fa-turbo").transcribe(out)
Path("results") / f"long_{minutes:g}min.txt"
(Path("results") / f"long_{minutes:g}min.txt").write_text(r["text"], encoding="utf-8")
print("seconds", r["seconds"], "rtf", r["rtf"])
print("one long file:       ", wer_cer(" ".join(refs), r["text"]))
print("same sentences split:", wer_cer(refs, hyps))
print("LONG_DONE")
