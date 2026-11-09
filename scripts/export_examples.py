"""Collect example inputs and outputs into examples/, with audio next to its text.

    python -m scripts.export_examples

Sentences are every 15th of the 300 scored ones, not hand-picked, so they stay representative.
"""
import csv
import json
import shutil
from pathlib import Path

from pv.metrics import wer_cer
from pv.text import prepare_for_tts
from scripts.evaluate import PARAGRAPHS

OUT, RESULTS, SAMPLES = Path("examples"), Path("results"), Path("samples")
EVERY = 15


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel reads Persian correctly
        w = csv.DictWriter(f, rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def write_text(path, text):
    Path(path).write_text(text, encoding="utf-8")


def part1():
    long_dir, sent_dir = OUT / "part1_stt" / "long_clips", OUT / "part1_stt" / "test_sentences"
    long_dir.mkdir(parents=True)
    sent_dir.mkdir(parents=True)
    rows = []
    for wav in sorted(SAMPLES.glob("stt_*.wav")):
        ref = wav.with_suffix(".txt").read_text(encoding="utf-8")
        hyp = (RESULTS / f"{wav.stem}.fa-turbo.txt").read_text(encoding="utf-8")
        shutil.copy(wav, long_dir / wav.name)
        write_text(long_dir / f"{wav.stem}.reference.txt", ref)
        write_text(long_dir / f"{wav.stem}.transcript.txt", hyp)
        rows.append({"file": wav.name, **wer_cer(ref, hyp), "reference": ref, "transcript": hyp})
    write_tsv(long_dir / "results.tsv", rows)

    with open(SAMPLES / "fleurs_test.tsv", encoding="utf-8") as f:
        paths = [r[0] for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    with open(RESULTS / "fleurs.fa-turbo.tsv", encoding="utf-8") as f:
        scored = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = []
    for i in range(0, len(scored), EVERY):
        name, ref, hyp = scored[i]
        assert Path(paths[i]).name == name
        stem = f"sentence_{i + 1:03d}"
        shutil.copy(paths[i], sent_dir / f"{stem}.wav")
        write_text(sent_dir / f"{stem}.reference.txt", ref)
        write_text(sent_dir / f"{stem}.transcript.txt", hyp)
        rows.append({"file": f"{stem}.wav", "source": name, **wer_cer(ref, hyp), "reference": ref, "transcript": hyp})
    write_tsv(sent_dir / "results.tsv", rows)
    return len(rows)


def part2():
    root = OUT / "part2_cloning"
    root.mkdir(parents=True)
    for i, text in enumerate(PARAGRAPHS, 1):
        write_text(root / f"paragraph_{i}.input.txt", text)
    final = json.loads((RESULTS / "eval.json").read_text(encoding="utf-8"))["tts"]
    rows = []
    for r in final["rows"]:
        stem = r["voice"].removesuffix(".wav")
        d = root / stem
        d.mkdir(exist_ok=True)
        shutil.copy(SAMPLES / r["voice"], d / "reference_voice.wav")
        shutil.copy(SAMPLES / f"{stem}.txt", d / "reference_voice.text.txt")
        out = f"cloned_paragraph_{r['paragraph']}.wav"
        shutil.copy(RESULTS / "clones_takes1" / f"{stem}_p{r['paragraph']}.wav", d / out)
        rows.append({"file": f"{stem}/{out}", "input": f"paragraph_{r['paragraph']}.input.txt",
                     "audio_seconds": r["audio_seconds"], "similarity": r["similarity"],
                     "roundtrip_wer": r["roundtrip"]["wer"], "roundtrip_cer": r["roundtrip"]["cer"],
                     "text_as_spoken": prepare_for_tts(PARAGRAPHS[r["paragraph"] - 1])})
    write_tsv(root / "results.tsv", rows)
    return len(rows)


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    print("part 1 test sentences:", part1())
    print("part 2 cloned clips:", part2())
