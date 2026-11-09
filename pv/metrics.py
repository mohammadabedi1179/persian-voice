"""Evaluation metrics: WER/CER for STT, speaker similarity for voice cloning."""
import jiwer
import numpy as np
import torch

from .audio import load_audio
from .text import normalize_for_scoring


def wer_cer(reference: str | list[str], hypothesis: str | list[str]) -> dict:
    """Corpus-level WER, CER and space-insensitive CER after Persian normalisation."""
    refs = [reference] if isinstance(reference, str) else reference
    hyps = [hypothesis] if isinstance(hypothesis, str) else hypothesis
    refs = [normalize_for_scoring(r) for r in refs]
    hyps = [normalize_for_scoring(h) for h in hyps]
    return {
        "wer": round(jiwer.wer(refs, hyps), 4),
        "cer": round(jiwer.cer(refs, hyps), 4),
        # Persian spacing is inconsistent (compound words), so also report CER without spaces
        "cer_nospace": round(jiwer.cer([r.replace(" ", "") for r in refs],
                                       [h.replace(" ", "") for h in hyps]), 4),
    }


def roundtrip_scorer(transcriber):
    """scorer(wav, sr, sentence) -> CER of what STT hears; extra babbled syllables raise it."""
    import librosa

    def score(wav, sr, sentence):
        heard = transcriber.transcribe_array(librosa.resample(wav, orig_sr=sr, target_sr=16000))
        return wer_cer(sentence, heard)["cer"]

    return score


class SpeakerSimilarity:
    """Cosine similarity of ECAPA-TDNN speaker embeddings (VoxCeleb-trained)."""

    def __init__(self, device: str | None = None):
        from speechbrain.inference.speaker import EncoderClassifier
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device},
            savedir="models/spkrec-ecapa-voxceleb")

    def embed(self, path: str) -> np.ndarray:
        wav = torch.from_numpy(load_audio(path, 16000)).unsqueeze(0)
        with torch.no_grad():
            e = self.enc.encode_batch(wav).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    def __call__(self, a: str, b: str) -> float:
        return round(float(self.embed(a) @ self.embed(b)), 4)
