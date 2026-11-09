"""Build test material from Google FLEURS (fa_ir, test split, CC-BY 4.0).

Writes to samples/:
  stt_XX.wav + stt_XX.txt   30–60 s Persian clips with exact reference text
  voice_XX.wav              6–15 s single-speaker clips used as cloning references
  fleurs_test.tsv           the full test list (for the corpus-level WER run)
"""
import csv
import random
import tarfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from pv.audio import load_audio, save_wav

SR = 16000
OUT = Path("samples")
REPO = "google/fleurs"


def main(n_stt=3, n_voices=4, seed=7):
    OUT.mkdir(exist_ok=True)
    tsv = hf_hub_download(REPO, "data/fa_ir/test.tsv", repo_type="dataset")
    tar = hf_hub_download(REPO, "data/fa_ir/audio/test.tar.gz", repo_type="dataset")
    audio_dir = Path("data/fleurs_fa_test")
    if not audio_dir.exists():
        with tarfile.open(tar) as t:
            t.extractall(audio_dir, filter="data")
    files = {p.name: p for p in audio_dir.rglob("*.wav")}

    # columns: id, file, raw transcription, normalized transcription, chars, n_samples, gender
    rows = []
    with open(tsv, encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r[1] in files:
                rows.append({"file": str(files[r[1]]), "text": r[2], "secs": int(r[5]) / SR,
                             "gender": r[6]})
    with open(OUT / "fleurs_test.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for r in rows:
            w.writerow([r["file"], r["text"], f"{r['secs']:.2f}", r["gender"]])
    print(f"{len(rows)} FLEURS fa test utterances")

    rng = random.Random(seed)
    pool = rows[:]
    rng.shuffle(pool)

    # 30–60 s STT samples: concatenate utterances (0.4 s gaps) until ≥ 40 s
    gap = np.zeros(int(0.4 * SR), np.float32)
    for k in range(1, n_stt + 1):
        wavs, texts, total = [], [], 0.0
        while total < 40 and pool:
            r = pool.pop()
            if total + r["secs"] > 60:
                continue
            wavs += [load_audio(r["file"], SR), gap]
            texts.append(r["text"])
            total += r["secs"] + 0.4
        save_wav(OUT / f"stt_{k:02d}.wav", np.concatenate(wavs[:-1]), SR)
        (OUT / f"stt_{k:02d}.txt").write_text(" ".join(texts), encoding="utf-8")
        print(f"stt_{k:02d}: {total:.1f} s, {len(texts)} utterances")

    # cloning references: single utterances of 8–15 s (the fa test split is all
    # male and has no speaker ids, so voices are picked at random)
    refs = [r for r in pool if 8 <= r["secs"] <= 15][:n_voices]
    for k, r in enumerate(refs, 1):
        save_wav(OUT / f"voice_{k:02d}.wav", load_audio(r["file"], SR), SR)
        (OUT / f"voice_{k:02d}.txt").write_text(r["text"], encoding="utf-8")
        print(f"voice_{k:02d}: {r['secs']:.1f} s")


if __name__ == "__main__":
    main()
