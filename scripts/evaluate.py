"""Measure both parts of the system and write results/eval.json + results/eval.md.

    python -m scripts.evaluate --stt-corpus 300      # STT only needs ungated models
    python -m scripts.evaluate --tts                 # + voice cloning (needs HF token)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

from pv.audio import save_wav
from pv.metrics import SpeakerSimilarity, roundtrip_scorer, wer_cer
from pv.stt import MODELS as STT_MODELS, Transcriber

SAMPLES, RESULTS = Path("samples"), Path("results")

# new paragraphs that appear in none of the reference recordings
PARAGRAPHS = [
    "پروژه ساختمان مسکونی در ۱۲ طبقه طراحی شده و عملیات اجرایی آن از بهار سال آینده آغاز می‌شود. "
    "برآورد اولیه نشان می‌دهد که حدود ۸۵۰۰ متر مکعب بتن برای اسکلت لازم است.",
    "صبح امروز هوای تهران نسبتاً پاک بود و بسیاری از مردم برای پیاده‌روی به پارک‌ها رفتند. "
    "کارشناسان هواشناسی پیش‌بینی می‌کنند که از فردا دما کاهش پیدا کند.",
    "هوش مصنوعی در سال‌های اخیر پیشرفت چشمگیری داشته است. "
    "امروزه مدل‌های زبانی می‌توانند متن بنویسند، ترجمه کنند و حتی به پرسش‌های تخصصی پاسخ دهند.",
]


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0


def eval_stt(corpus_n: int) -> dict:
    res = {}
    long_clips = sorted(SAMPLES.glob("stt_*.wav"))
    with open(SAMPLES / "fleurs_test.tsv", encoding="utf-8") as f:
        corpus = [r for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)][:corpus_n]
    for name in STT_MODELS:
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        tr = Transcriber(name)
        load_s = round(time.perf_counter() - t0, 1)
        clips = []
        for wav in long_clips:
            ref = wav.with_suffix(".txt").read_text(encoding="utf-8")
            r = tr.transcribe(str(wav))
            (RESULTS / f"{wav.stem}.{name}.txt").write_text(r["text"], encoding="utf-8")
            clips.append({"clip": wav.name, **{k: r[k] for k in ("audio_seconds", "seconds", "rtf")},
                          **wer_cer(ref, r["text"])})
            print(name, clips[-1])
        refs, hyps, secs, audio = [], [], 0.0, 0.0
        for path, ref, *_ in corpus:
            r = tr.transcribe(path)
            refs.append(ref), hyps.append(r["text"])
            secs += r["seconds"]
            audio += r["audio_seconds"]
        with open(RESULTS / f"fleurs.{name}.tsv", "w", encoding="utf-8", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(
                (Path(p).name, ref, hyp) for (p, *_), ref, hyp in zip(corpus, refs, hyps))
        res[name] = {"model": STT_MODELS[name], "load_seconds": load_s, "long_clips": clips,
                     "corpus": {"utterances": len(refs), "audio_minutes": round(audio / 60, 1),
                                "rtf": round(secs / max(audio, 1e-6), 3), **wer_cer(refs, hyps)},
                     "peak_vram_gb": peak_gb()}
        print(name, res[name]["corpus"])
        del tr
        torch.cuda.empty_cache()
    return res


def eval_tts(model_name: str, takes: int = 1) -> dict:
    from pv.tts import VoiceCloner
    out_dir = RESULTS / f"clones_takes{takes}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    cl = VoiceCloner(model_name)
    sim = SpeakerSimilarity()
    asr = Transcriber("fa-turbo")  # round trip: can the generated speech be understood?
    voices = sorted(SAMPLES.glob("voice_*.wav"))
    rows = []
    for v in voices:
        for i, text in enumerate(PARAGRAPHS, 1):
            torch.manual_seed(1)
            r = cl.clone(str(v), text, takes=takes, scorer=roundtrip_scorer(asr))
            out = save_wav(out_dir / f"{v.stem}_p{i}.wav", r["wav"], r["sr"])
            back = asr.transcribe(out)["text"]
            rows.append({"voice": v.name, "paragraph": i, "audio_seconds": r["audio_seconds"],
                         "seconds": r["seconds"], "rtf": r["rtf"],
                         "similarity": sim(str(v), out),
                         "roundtrip": wer_cer(text_for_scoring(text), back)})
            print(rows[-1])
    # context for the similarity number: the same metric between different real speakers
    cross = [sim(str(a), str(b)) for i, a in enumerate(voices) for b in voices[i + 1:]]
    # voice_02 and paragraph 3 were not used when choosing the settings in scripts/tune_clone.py
    held_out = [r for r in rows if r["voice"] == "voice_02.wav" or r["paragraph"] == 3]
    return {"model": model_name, "takes": takes, "rows": rows, "peak_vram_gb": peak_gb(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "mean": means(rows), "held_out_mean": means(held_out), "held_out_n": len(held_out),
            "between_real_speakers_similarity": round(sum(cross) / max(len(cross), 1), 4)}


def means(rows):
    n = len(rows)
    return {"similarity": round(sum(r["similarity"] for r in rows) / n, 4),
            "rtf": round(sum(r["rtf"] for r in rows) / n, 3),
            "roundtrip_wer": round(sum(r["roundtrip"]["wer"] for r in rows) / n, 4),
            "roundtrip_cer": round(sum(r["roundtrip"]["cer"] for r in rows) / n, 4)}


def text_for_scoring(text):
    from pv.text import prepare_for_tts
    return prepare_for_tts(text)  # digits spelled out, as they are spoken


def write_md(res: dict):
    L = ["# Evaluation results", ""]
    if "stt" in res:
        L += ["## Speech-to-text", "", "| model | corpus WER | CER | CER (no space) | RTF | VRAM GB |",
              "|---|---|---|---|---|---|"]
        for name, r in res["stt"].items():
            c = r["corpus"]
            L.append(f"| {name} | {c['wer']:.1%} | {c['cer']:.1%} | {c['cer_nospace']:.1%} | {c['rtf']} | {r['peak_vram_gb']} |")
        any_r = next(iter(res["stt"].values()))["corpus"]
        L += ["", f"Corpus: {any_r['utterances']} FLEURS fa test utterances ({any_r['audio_minutes']} min).", "",
              "| clip | model | seconds | WER | CER |", "|---|---|---|---|---|"]
        for name, r in res["stt"].items():
            for c in r["long_clips"]:
                L.append(f"| {c['clip']} ({c['audio_seconds']} s) | {name} | {c['seconds']} | {c['wer']:.1%} | {c['cer']:.1%} |")
    for key, title in [("tts_before_fixes", "Voice cloning, before fixes"),
                       ("tts", "Voice cloning, final settings, one take"),
                       ("tts_retake", "Voice cloning, final settings, up to 3 takes per sentence")]:
        if key not in res:
            continue
        t = res[key]
        m = t["mean"]
        L += ["", f"## {title}", "",
              f"Model `{t['model']}`, {len(t['rows'])} generations, device {t.get('device', 'cuda')}, peak VRAM {t['peak_vram_gb']} GB.", "",
              f"- mean speaker similarity (reference vs clone): **{m['similarity']:.2f}**",
              f"- between reference clips (FLEURS has no speaker ids, so some may be one person): {t['between_real_speakers_similarity']:.2f}",
              f"- round-trip WER / CER (clone -> STT): {m['roundtrip_wer']:.1%} / {m['roundtrip_cer']:.1%}",
              f"- mean RTF: {m['rtf']}"]
        if "held_out_mean" in t:
            h = t["held_out_mean"]
            L.append(f"- held out from tuning ({t['held_out_n']} clips: voice_02 or paragraph 3): "
                     f"WER / CER {h['roundtrip_wer']:.1%} / {h['roundtrip_cer']:.1%}, similarity {h['similarity']:.2f}")
        L += ["", "| voice | paragraph | audio s | RTF | similarity | round-trip CER |", "|---|---|---|---|---|---|"]
        for r in t["rows"]:
            L.append(f"| {r['voice']} | {r['paragraph']} | {r['audio_seconds']} | {r['rtf']} | {r['similarity']:.2f} | {r['roundtrip']['cer']:.1%} |")
    (RESULTS / "eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt-corpus", type=int, default=300)
    ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--tts-model", default="parsvoice-xtts")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "eval.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not a.no_stt:
        res["stt"] = eval_stt(a.stt_corpus)
    if a.tts:
        if "tts" in res and "takes" not in res["tts"]:  # keep the first run as the before/after baseline
            res["tts_before_fixes"] = res["tts"]
        res["tts"] = eval_tts(a.tts_model, 1)
        res["tts_retake"] = eval_tts(a.tts_model, 3)
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(res)
    print((RESULTS / "eval.md").read_text(encoding="utf-8"))
