"""Gradio UI: Persian speech-to-text and voice cloning.

    .venv/bin/python app.py            # http://<server>:7860
"""
import argparse
import tempfile
from functools import lru_cache

import gradio as gr
import torch

from pv.audio import load_audio, save_wav
from pv.metrics import SpeakerSimilarity, wer_cer
from pv.stt import MODELS as STT_MODELS, Transcriber
from pv.tts import VoiceCloner

CONSENT = "صاحب این صدا به استفاده از آن برای شبیه‌سازی رضایت داده است."
CSS = ".rtl textarea, .rtl input, .rtl .prose {direction: rtl; text-align: right;}"


@lru_cache(maxsize=None)
def transcriber(name):
    return Transcriber(name)


@lru_cache(maxsize=None)
def cloner():
    return VoiceCloner("parsvoice-xtts")


@lru_cache(maxsize=None)
def similarity():
    return SpeakerSimilarity()


def run_stt(audio, model_name, reference):
    if not audio:
        raise gr.Error("یک فایل صوتی بارگذاری کنید.")
    r = transcriber(model_name).transcribe(audio)
    lines = [f"- مدت صوت: **{r['audio_seconds']} ثانیه** — زمان پردازش: **{r['seconds']} ثانیه** (RTF {r['rtf']})"]
    if reference and reference.strip():
        m = wer_cer(reference, r["text"])
        lines.append(f"- WER: **{m['wer']:.1%}** — CER: **{m['cer']:.1%}** — CER بدون فاصله: **{m['cer_nospace']:.1%}**")
    return r["text"], "\n".join(lines), audio


def run_clone(reference_audio, text, consent, temperature):
    if not consent:
        raise gr.Error("بدون رضایت صاحب صدا، شبیه‌سازی انجام نمی‌شود.")
    if not reference_audio:
        raise gr.Error("نمونه صدای گوینده را بارگذاری کنید.")
    if not text or not text.strip():
        raise gr.Error("متن موردنظر را وارد کنید.")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    seconds = len(load_audio(reference_audio, 16000)) / 16000
    if not 6 <= seconds <= 30:  # the model card recommends roughly 6 to 15 s
        gr.Warning(f"طول نمونه صدا {seconds:.0f} ثانیه است؛ ۶ تا ۱۵ ثانیه پیشنهاد می‌شود.")
    r = cloner().clone(reference_audio, text, temperature)
    out = save_wav(tempfile.mktemp(suffix=".wav"), r["wav"], r["sr"])
    sim = similarity()(reference_audio, out)
    vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    info = (f"- مدت صوت تولیدی: **{r['audio_seconds']} ثانیه** — زمان تولید: **{r['seconds']} ثانیه** (RTF {r['rtf']})\n"
            f"- شباهت گوینده (کسینوس ECAPA): **{sim:.2f}** (بازه ۱- تا ۱؛ هرچه بیشتر، شبیه‌تر)\n"
            f"- بیشینه حافظه GPU: **{vram:.1f} GB**")
    return out, info


def build():
    with gr.Blocks(title="پردازش گفتار فارسی") as demo:
        gr.Markdown("# سامانه پردازش گفتار فارسی و شبیه‌سازی صدا", elem_classes="rtl")
        with gr.Tab("۱. گفتار به متن"):
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(label="فایل صوتی فارسی (۳۰ تا ۶۰ ثانیه)", type="filepath")
                    model = gr.Dropdown(list(STT_MODELS), value="fa-turbo", label="مدل")
                    ref = gr.Textbox(label="متن مرجع (اختیاری، برای محاسبه WER)", lines=4,
                                     rtl=True, elem_classes="rtl")
                    btn = gr.Button("تبدیل به متن", variant="primary")
                with gr.Column():
                    hyp = gr.Textbox(label="متن استخراج‌شده", lines=10, rtl=True,
                                     buttons=["copy"], elem_classes="rtl")
                    stt_info = gr.Markdown(elem_classes="rtl")
        with gr.Tab("۲. متن به گفتار با صدای گوینده"):
            with gr.Row():
                with gr.Column():
                    ref_audio = gr.Audio(label="نمونه صدای گوینده (۶ تا ۱۵ ثانیه، بدون نویز)", type="filepath")
                    text = gr.Textbox(label="پاراگراف جدید", lines=6, rtl=True, elem_classes="rtl")
                    consent = gr.Checkbox(label=CONSENT)
                    temp = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="دما (تنوع بیان)")
                    btn2 = gr.Button("تولید صدا", variant="primary")
                with gr.Column():
                    out_audio = gr.Audio(label="صدای تولیدشده", type="filepath")
                    tts_info = gr.Markdown(elem_classes="rtl")
        # the audio uploaded for STT also becomes the default voice sample
        btn.click(run_stt, [audio, model, ref], [hyp, stt_info, ref_audio])
        btn2.click(run_clone, [ref_audio, text, consent, temp], [out_audio, tts_info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="temporary public gradio.live link")
    a = ap.parse_args()
    build().queue().launch(server_name=a.host, server_port=a.port, share=a.share,
                           css=CSS, theme=gr.themes.Soft())
