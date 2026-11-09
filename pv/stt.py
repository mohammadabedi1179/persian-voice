"""Persian speech-to-text with Whisper (Hugging Face transformers pipeline)."""
import time

import numpy as np
import torch
from transformers import pipeline

from .audio import load_audio

MODELS = {
    # Whisper large-v3-turbo fine-tuned on ~534 h of Persian (MIT); CV17-fa WER 19.4%
    "fa-turbo": "MohammadGholizadeh/whisper-large-v3-turbo-persian",
    # the untouched base model, kept as the baseline for the report
    "base-turbo": "openai/whisper-large-v3-turbo",
}
SR = 16000


def split_on_pauses(wav: np.ndarray, max_s: float = 28.0, min_s: float = 12.0) -> list[np.ndarray]:
    """Cut audio longer than Whisper's 30 s window at the quietest point.

    The pipeline's own overlapping-chunk merge can drop words that straddle a
    window edge (seen on a 41 s clip); cutting in a pause avoids that. Each cut
    is placed at the lowest-energy 30 ms frame between min_s and max_s.
    """
    hop = int(0.03 * SR)
    segs, start = [], 0
    while len(wav) - start > max_s * SR:
        lo, hi = start + int(min_s * SR), start + int(max_s * SR)
        frames = wav[lo:hi][: (hi - lo) // hop * hop].reshape(-1, hop)
        cut = lo + int(np.argmin((frames ** 2).mean(axis=1))) * hop + hop // 2
        segs.append(wav[start:cut])
        start = cut
    segs.append(wav[start:])
    return segs


class Transcriber:
    def __init__(self, name: str = "fa-turbo", device: str | None = None):
        self.name = name
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=MODELS.get(name, name),
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device=device,
        )

    def transcribe_array(self, wav: np.ndarray) -> str:
        """Transcribe a mono 16 kHz clip already in memory."""
        outs = self.pipe([{"raw": s, "sampling_rate": SR} for s in split_on_pauses(wav)], batch_size=8,
                         generate_kwargs={"language": "fa", "task": "transcribe"})
        return " ".join(o["text"].strip() for o in outs)

    def transcribe(self, path: str) -> dict:
        wav = load_audio(path, SR)
        t0 = time.perf_counter()
        segments = [{"raw": s, "sampling_rate": SR} for s in split_on_pauses(wav)]
        outs = self.pipe(segments, batch_size=8,
                         generate_kwargs={"language": "fa", "task": "transcribe"})
        secs = time.perf_counter() - t0
        audio_secs = len(wav) / SR
        return {
            "text": " ".join(o["text"].strip() for o in outs),
            "audio_seconds": round(audio_secs, 2),
            "seconds": round(secs, 2),
            "rtf": round(secs / max(audio_secs, 1e-6), 3),
        }
