"""Compare cloning settings with the same voices, texts and seeds; write results/tune.md.

    python -m scripts.tune_clone          # runs on whatever device is visible
"""
import argparse
import csv
from pathlib import Path

import torch

from pv.audio import save_wav
from pv.metrics import SpeakerSimilarity, roundtrip_scorer, wer_cer
from pv.stt import Transcriber
from pv.text import prepare_for_tts
from pv.tts import VoiceCloner
from scripts.evaluate import PARAGRAPHS

OUT = Path("results/tune")
VOICES = ["voice_01", "voice_03", "voice_04"]
TEXTS = PARAGRAPHS[:2]
SEEDS = [1, 2]
VARIANTS = {
    "A_before_fixes": dict(temperature=0.7, normalize_ref=False, config_cond=False),
    "B_cond+loudness": dict(temperature=0.7),
    "C_B+temp0.5": dict(temperature=0.5),
    "D_B+no_final_punct": dict(temperature=0.7, strip_final_punct=True),
    "E_B+temp0.5+no_punct": dict(temperature=0.5, strip_final_punct=True),
    "F_B+3takes": dict(temperature=0.7, takes=3),
    "G_B+temp0.5+3takes": dict(temperature=0.5, takes=3),
}


def main(names, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    cl, sim, asr = VoiceCloner("parsvoice-xtts"), SpeakerSimilarity(), Transcriber("fa-turbo")
    scorer = roundtrip_scorer(asr)
    rows = []
    for name in names:
        kw = dict(VARIANTS[name], scorer=scorer)
        for v in VOICES:
            ref = f"samples/{v}.wav"
            for i, text in enumerate(TEXTS, 1):
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    r = cl.clone(ref, text, **kw)
                    wav = save_wav(OUT / f"{name}.{v}.p{i}.s{seed}.wav", r["wav"], r["sr"])
                    heard = asr.transcribe(wav)["text"]
                    m = wer_cer(prepare_for_tts(text), heard)
                    rows.append({"variant": name, "voice": v, "text": i, "seed": seed, "seconds": r["audio_seconds"],
                                 "rtf": r["rtf"], "similarity": sim(ref, wav), **m, "heard": heard})
                    print({k: x for k, x in rows[-1].items() if k != "heard"}, flush=True)
    with open(f"results/{tag}.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    avg = lambda rs, k: sum(r[k] for r in rs) / len(rs)
    L = [f"{len(VOICES)} voices x {len(TEXTS)} texts x {len(SEEDS)} seeds per variant.", "",
         "| variant | voice | round-trip WER | round-trip CER | similarity | audio s |", "|---|---|---|---|---|---|"]
    for name in names:
        for v in VOICES + ["ALL"]:
            rs = [r for r in rows if r["variant"] == name and v in (r["voice"], "ALL")]
            L.append(f"| {name} | {v} | {avg(rs, 'wer'):.1%} | {avg(rs, 'cer'):.1%} | {avg(rs, 'similarity'):.2f} | {avg(rs, 'seconds'):.1f} |")
    Path(f"results/{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("TUNE_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[n for n in VARIANTS if n[0] in "ABCDE"])
    ap.add_argument("--tag", default="tune")
    a = ap.parse_args()
    main(a.variants, a.tag)
