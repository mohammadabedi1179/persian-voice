"""Audio I/O: decode any format (wav, mp3, m4a, ogg, ...) to mono float32 at a given rate."""
import subprocess

import numpy as np
import soundfile as sf


def _ffmpeg() -> str:
    import imageio_ffmpeg  # bundled static ffmpeg; the server has no system ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-i", str(path),
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def save_wav(path: str, wav: np.ndarray, sr: int) -> str:
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sr)
    return str(path)


def duration(wav: np.ndarray, sr: int) -> float:
    return len(wav) / sr


def normalize_loudness(wav: np.ndarray, sr: int, target_db: float = -23.0) -> np.ndarray:
    """Scale so loud speech frames (90th percentile of 30 ms RMS) sit at target_db."""
    hop = int(0.03 * sr)
    rms = np.sqrt((wav[: len(wav) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1) + 1e-12)
    gain = 10 ** (target_db / 20) / max(np.percentile(rms, 90), 1e-6)
    peak = np.abs(wav).max() * gain
    return wav * (gain if peak <= 0.97 else gain * 0.97 / peak)
