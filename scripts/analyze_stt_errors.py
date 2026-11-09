"""Break down the saved FLEURS transcripts and write results/stt_errors.md.

    python -m scripts.analyze_stt_errors        # needs only results/fleurs.*.tsv
"""
import collections
import csv
import random
from pathlib import Path

import jiwer
import numpy as np

from pv.text import normalize_for_scoring

RESULTS = Path("results")
NAMES = ["fa-turbo", "base-turbo"]
RESAMPLES = 2000


def analyse(name):
    with open(RESULTS / f"fleurs.{name}.tsv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    refs = [normalize_for_scoring(r[1]) for r in rows]
    hyps = [normalize_for_scoring(r[2]) for r in rows]
    out = jiwer.process_words(refs, hyps)
    edits, lengths, subs = [], [], collections.Counter()
    for ref, hyp, chunks in zip(out.references, out.hypotheses, out.alignments):
        edits.append(sum(max(c.ref_end_idx - c.ref_start_idx, c.hyp_end_idx - c.hyp_start_idx)
                         for c in chunks if c.type != "equal"))
        lengths.append(len(ref))
        for c in chunks:
            if c.type == "substitute":
                subs.update(zip(ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]))
    return {"s": out.substitutions, "d": out.deletions, "i": out.insertions, "hits": out.hits,
            "edits": np.array(edits), "lengths": np.array(lengths), "subs": subs}


def bootstrap(fn, n, seed=7):
    """Sentence-level percentile interval of fn(indices)."""
    rng = random.Random(seed)
    vals = sorted(fn([rng.randrange(n) for _ in range(n)]) for _ in range(RESAMPLES))
    return vals[int(0.025 * RESAMPLES)], vals[int(0.975 * RESAMPLES)]


def main():
    res = {name: analyse(name) for name in NAMES}
    n = len(res["fa-turbo"]["edits"])
    wer = lambda r, idx: r["edits"][idx].sum() / r["lengths"][idx].sum()
    L = ["| model | reference words | correct | S | D | I | WER | 95% interval | sentences with no error | median sentence WER | P90 sentence WER |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        per = r["edits"] / r["lengths"]
        lo, hi = bootstrap(lambda idx: wer(r, idx), n)
        L.append(f"| {name} | {r['lengths'].sum()} | {r['hits']} | {r['s']} | {r['d']} | {r['i']} | "
                 f"{wer(r, list(range(n))):.2%} | {lo:.2%} to {hi:.2%} | {(r['edits'] == 0).sum()} of {n} | "
                 f"{np.median(per):.1%} | {np.percentile(per, 90):.1%} |")
    fa, base = res["fa-turbo"], res["base-turbo"]
    lo, hi = bootstrap(lambda idx: wer(base, idx) - wer(fa, idx), n)
    L += ["", f"WER difference, base minus fa, same sentences: 95% interval {lo:.2%} to {hi:.2%} ({RESAMPLES} resamples).",
          "", "| reference word | fa-turbo output | count |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for (a, b), c in fa["subs"].most_common(12)]
    (RESULTS / "stt_errors.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
