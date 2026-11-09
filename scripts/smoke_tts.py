"""Quick check that the cloning path loads and synthesises: python -m scripts.smoke_tts [model]."""
import sys

from pv.audio import save_wav
from pv.tts import VoiceCloner

name = sys.argv[1] if len(sys.argv) > 1 else "parsvoice-xtts"
cl = VoiceCloner(name)
r = cl.clone("samples/voice_01.wav", "سلام. این یک آزمایش کوتاه برای بررسی سامانه شبیه‌سازی صدا است.")
save_wav(f"results/smoke_{name}.wav", r["wav"], r["sr"])
print({k: v for k, v in r.items() if k != "wav"})
