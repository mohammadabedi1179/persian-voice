"""Zero-shot Persian voice cloning with XTTS-v2 fine-tuned on ParsVoice."""
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from .audio import load_audio, normalize_loudness, save_wav
from .text import prepare_for_tts, split_sentences

MODELS = {
    # XTTS-v2 fine-tuned on ParsVoice (2,200 h, 1,800+ Persian speakers); gated, needs HF token
    "parsvoice-xtts": ("MohammadJRanjbar/ParsVoice-XTTS", "fa"),
    # original multilingual XTTS-v2 (no Persian); only used to test the plumbing
    "xtts-v2": ("coqui/XTTS-v2", "ar"),
}
GAP_SECONDS = 0.18  # silence between synthesised sentences (chosen by hand, not tuned)
GOOD_ENOUGH_CER = 0.05  # retake above this round-trip error (chosen by hand, not tuned)


def _allow_persian_in_tokenizer(tokenizer):
    """Stock XTTS rejects language codes it was not released with ('fa').

    Persian text needs no cleaner beyond what prepare_for_tts already did, so
    let 'fa' through unchanged and defer to the original for everything else.
    """
    original = tokenizer.preprocess_text

    def preprocess_text(txt, lang):
        if lang == "fa":
            return " ".join(txt.split())
        return original(txt, lang)

    tokenizer.preprocess_text = preprocess_text


class VoiceCloner:
    def __init__(self, name: str = "parsvoice-xtts", device: str | None = None):
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        repo, self.lang = MODELS[name]
        self.name = name
        ckpt = Path(f"models/{repo.split('/')[-1]}")
        if not (ckpt / "model.pth").exists():  # once downloaded, no HF token is needed
            snapshot_download(repo, local_dir=ckpt, allow_patterns=["*.json", "*.pth"])
        cfg = XttsConfig()
        cfg.load_json(str(ckpt / "config.json"))
        self.model = Xtts.init_from_config(cfg)
        self.model.load_checkpoint(cfg, checkpoint_dir=str(ckpt), use_deepspeed=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        _allow_persian_in_tokenizer(self.model.tokenizer)
        self.cfg = cfg
        self.sr = cfg.audio.output_sample_rate  # 24 kHz

    def _latents(self, reference_wav: str, normalize_ref: bool, config_cond: bool):
        """Speaker latents from the reference; a very quiet clip degrades them badly."""
        path = reference_wav
        with tempfile.TemporaryDirectory() as tmp:
            if normalize_ref:
                path = save_wav(Path(tmp) / "ref.wav", normalize_loudness(load_audio(reference_wav, 24000), 24000), 24000)
            kw = {}
            if config_cond:  # the settings this checkpoint ships with, not the library defaults
                kw = dict(gpt_cond_len=self.cfg.gpt_cond_len, gpt_cond_chunk_len=self.cfg.gpt_cond_chunk_len,
                          max_ref_length=self.cfg.max_ref_len, sound_norm_refs=self.cfg.sound_norm_refs)
            return self.model.get_conditioning_latents(audio_path=[path], **kw)

    @torch.inference_mode()
    def clone(self, reference_wav: str, text: str, temperature: float = 0.5, normalize_ref: bool = True,
              config_cond: bool = True, strip_final_punct: bool = True, takes: int = 1, scorer=None) -> dict:
        """Defaults come from results/tune.md: final punctuation makes this checkpoint babble on.

        With takes > 1 and scorer(wav, sr, sentence) -> error, a sentence is regenerated
        only while its error exceeds GOOD_ENOUGH_CER, and the lowest-error take is kept.
        """
        t0 = time.perf_counter()
        latent, spk = self._latents(reference_wav, normalize_ref, config_cond)
        gap = np.zeros(int(GAP_SECONDS * self.sr), dtype=np.float32)
        pieces = []
        for sentence in split_sentences(prepare_for_tts(text)):
            if strip_final_punct:
                sentence = re.sub(r"[.!?؟؛،,]+$", "", sentence)
            best, best_error = None, float("inf")
            for _ in range(takes if scorer else 1):
                take = np.asarray(self.model.inference(
                    sentence, self.lang, latent, spk, temperature=temperature,
                    repetition_penalty=5.0, enable_text_splitting=False)["wav"], dtype=np.float32)
                error = scorer(take, self.sr, sentence) if scorer and takes > 1 else 0.0
                if error < best_error:
                    best, best_error = take, error
                if best_error <= GOOD_ENOUGH_CER:
                    break
            pieces += [best, gap]
        wav = np.concatenate(pieces[:-1]) if pieces else np.zeros(1, np.float32)
        secs = time.perf_counter() - t0
        return {
            "wav": wav, "sr": self.sr,
            "audio_seconds": round(len(wav) / self.sr, 2),
            "seconds": round(secs, 2),
            "rtf": round(secs / max(len(wav) / self.sr, 1e-6), 3),
        }
