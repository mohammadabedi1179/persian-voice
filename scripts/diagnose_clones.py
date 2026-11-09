"""Inspect cloned audio on CPU: reference clip quality and what STT hears in each clone.

    python -m scripts.diagnose_clones
"""
from pathlib import Path

import numpy as np

from pv.audio import load_audio
from pv.stt import Transcriber
from scripts.evaluate import PARAGRAPHS

SR = 16000
asr = Transcriber("fa-turbo", device="cpu")


def level(path):
    """Overall level and a rough noise floor (quietest 10% of 30 ms frames), in dB."""
    wav = load_audio(path, SR)
    hop = int(0.03 * SR)
    rms = np.sqrt((wav[: len(wav) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1) + 1e-12)
    speech, floor = np.percentile(rms, 90), np.percentile(rms, 10)
    return (f"{len(wav) / SR:.1f} s, speech {20 * np.log10(speech):.1f} dB, "
            f"floor {20 * np.log10(floor):.1f} dB, SNR~{20 * np.log10(speech / floor):.1f} dB")


for v in sorted(Path("samples").glob("voice_*.wav")):
    print(f"\n=== {v.name}: {level(v)}")
    print("REF TEXT:", v.with_suffix(".txt").read_text(encoding="utf-8"))
    print("STT OF REF:", asr.transcribe(str(v))["text"])
    for i in (1, 2):
        c = Path("results/clones") / f"{v.stem}_p{i}.wav"
        print(f"--- p{i}: {level(c)}")
        print("INPUT:", PARAGRAPHS[i - 1])
        print("HEARD:", asr.transcribe(str(c))["text"])
