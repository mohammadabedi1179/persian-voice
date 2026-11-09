"""Rescore saved FLEURS transcripts under different ZWNJ rules (no model needed).

    python -m scripts.rescore
"""
import csv
from collections import Counter
from pathlib import Path

import jiwer

from pv import text as T

RULES = {"zwnj→space": " ", "zwnj→joined": "", "affix-canonical": None}


def norm(s, zwnj):
    if zwnj is None:
        return T.normalize_for_scoring(s)  # the rule used for the reported numbers
    s = T._NUMBER.sub(T._num_repl, T.unify(s)).replace("‌", zwnj)
    return T._SPACES.sub(" ", T._PUNCT.sub(" ", s)).strip()


for f in sorted(Path("results").glob("fleurs.*.tsv")):
    rows = list(csv.reader(open(f, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    print(f.name)
    for rule, z in RULES.items():
        refs = [norm(r[1], z) for r in rows]
        hyps = [norm(r[2], z) for r in rows]
        out = jiwer.process_words(refs, hyps)
        subs = Counter()
        for a, ref, hyp in zip(out.alignments, out.references, out.hypotheses):
            for c in a:
                if c.type == "substitute":
                    subs[(" ".join(ref[c.ref_start_idx:c.ref_end_idx]),
                          " ".join(hyp[c.hyp_start_idx:c.hyp_end_idx]))] += 1
        print(f"  {rule:12s} WER {out.wer:.2%}  S={out.substitutions} D={out.deletions} I={out.insertions}")
        print("   top substitutions:", subs.most_common(8))
