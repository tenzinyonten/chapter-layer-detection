#!/usr/bin/env python3
"""
Error analysis for a Chapter model from an eval_viterbi_iou.py --dump-spans file.

Usage:
    python src/analyze_chapter_errors.py --spans results/error_analysis/chapter_val_spans.jsonl \
        --pred-key viterbi_bp4.0 --tag val
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/tsawa_audit.csv"
CLEAN = ROOT / "data/chapter_spans_clean.csv"
TOKENIZER = "jhu-clsp/mmBERT-base"
PRE = re.compile(r"(^|\n)[༈༄༅།\s\d༠-༩\(\)\{\}\[\]༼༽\.]*$")
POST = re.compile(r"^[།༎་\s\xa0\]\)\}༽]*(\n|$)")


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0])) if inter else 0.0


def merge_overlaps(spans):
    out = []
    for s, e in sorted(spans):
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def match(gold, pred, thr=0.5):
    """Greedy one-to-one matching by IoU. Returns matched gold idx, matched pred idx."""
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gold) for j, p in enumerate(pred)
                    if iou(g, p) >= thr), reverse=True)
    mg, mp = set(), set()
    for _, i, j in pairs:
        if i not in mg and j not in mp:
            mg.add(i)
            mp.add(j)
    return mg, mp


def bucket(n):
    return "<15" if n < 15 else "15-29" if n < 30 else "30-59" if n < 60 else "60+"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spans", type=Path, required=True)
    ap.add_argument("--pred-key", default="viterbi_bp4.0")
    ap.add_argument("--tag", default="val")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/error_analysis")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    tok.model_max_length = int(1e12)
    audit = {r["pecha_id"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8"))}

    pred_tok = defaultdict(set)
    for line in args.spans.open(encoding="utf-8"):
        r = json.loads(line)
        for a, b in r[f"{args.pred_key}_tok"]:
            pred_tok[r["pecha_id"]].add((a, b))
    books = sorted(pred_tok.keys() | {json.loads(l)["pecha_id"] for l in args.spans.open(encoding="utf-8")})

    gold_all = defaultdict(list)
    for r in csv.DictReader(CLEAN.open(encoding="utf-8")):
        if r["dropped"] != "True":
            gold_all[r["pecha_id"]].append((int(r["start"]), int(r["end"])))

    rows_missed, rows_fp, per_book = [], [], []
    tot = Counter()
    miss_by = defaultdict(Counter)   # feature -> {"gold":n, "missed":n}
    for p in books:
        text = Path(audit[p]["base_path"]).read_text(encoding="utf-8")
        offs = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        pred = merge_overlaps([(offs[a][0], offs[b][1]) for a, b in pred_tok[p]
                               if 0 <= a <= b < len(offs)])
        gold = sorted(gold_all[p])
        mg, mp = match(gold, pred)
        tp, fn, fp = len(mg), len(gold) - len(mg), len(pred) - len(mp)
        per_book.append({"pecha_id": p, "batch": "old" if p.startswith("P") else "new",
                         "gold": len(gold), "pred": len(pred), "tp": tp, "fn": fn, "fp": fp})
        tot.update(tp=tp, fn=fn, fp=fp, gold=len(gold), pred=len(pred))
        for i, (s, e) in enumerate(gold):
            t = text[s:e]
            shaped = bool(PRE.search(text[max(0, s - 6):s]) and POST.match(text[e:e + 8]))
            feats = {"length " + bucket(e - s): 1,
                     "batch " + ("old" if p.startswith("P") else "new"): 1,
                     "has ལེའུ": int("ལེའུ" in t), "starts ༈": int(t.lstrip().startswith("༈")),
                     "starts with digit/bracket": int(bool(re.match(r"^\s*[\(\[\{༼]?\s*[\d༠-༩]", t))),
                     "book has <=2 gold spans": int(len(gold) <= 2)}
            for k, v in feats.items():
                if v:
                    miss_by[k]["gold"] += 1
                    miss_by[k]["missed"] += int(i not in mg)
            if i not in mg:
                rows_missed.append({"pecha_id": p, "start": s, "end": e, "len": e - s,
                                    "text": t[:120], "before": text[max(0, s - 20):s], "book_gold": len(gold)})
        for j, (s, e) in enumerate(pred):
            if j in mp:
                continue
            ov = max((iou((s, e), g) for g in gold), default=0.0)
            touches = any(min(e, g[1]) > max(s, g[0]) for g in gold)
            shaped = bool(PRE.search(text[max(0, s - 6):s]) and POST.match(text[e:e + 8]))
            rows_fp.append({"pecha_id": p, "start": s, "end": e, "len": e - s, "text": text[s:e][:120],
                            "best_iou": round(ov, 2), "touches_gold": touches,
                            "heading_shaped": shaped, "book_gold": len(gold)})

    P = tot["tp"] / max(tot["pred"], 1)
    R = tot["tp"] / max(tot["gold"], 1)
    F = 2 * P * R / max(P + R, 1e-9)
    print(f"{args.tag}: gold {tot['gold']}  pred {tot['pred']}  TP {tot['tp']}  "
          f"P {P:.3f}  R {R:.3f}  F1 {F:.3f}   (character-space, deduped windows)")

    print("\n== books by missed gold spans (worst 12) ==")
    for b in sorted(per_book, key=lambda x: -x["fn"])[:12]:
        print(f"  {b['pecha_id']:10} {b['batch']}  gold {b['gold']:3}  found {b['tp']:3}  missed {b['fn']:3}  fp {b['fp']:3}")
    zero = [b for b in per_book if b["gold"] and b["tp"] == 0]
    print(f"books with gold spans and zero found: {len(zero)} -> "
          f"{[(b['pecha_id'], b['gold']) for b in zero]}")

    print("\n== miss rate by kind of gold span ==")
    for k in sorted(miss_by):
        v = miss_by[k]
        print(f"  {k:28} gold {v['gold']:4}  missed {v['missed']:4}  ({100 * v['missed'] / v['gold']:.0f}%)")

    print("\n== false positives ==")
    n = len(rows_fp) or 1
    print(f"  total {len(rows_fp)}  touching no gold span {sum(not r['touches_gold'] for r in rows_fp)}  "
          f"heading-shaped {sum(r['heading_shaped'] for r in rows_fp)}  "
          f"in books with <=2 gold spans {sum(r['book_gold'] <= 2 for r in rows_fp)}  "
          f"partial overlap (IoU<0.5) {sum(r['touches_gold'] for r in rows_fp)}")

    def show(title, rows, extra):
        print(f"\n== {title} (first {args.show}) ==")
        for r in rows[:args.show]:
            print(f"  {r['pecha_id']} [{r['len']:3}] {extra(r)} {r['text'][:70]!r}")

    show("missed gold spans", sorted(rows_missed, key=lambda r: r["pecha_id"]),
         lambda r: f"(book gold {r['book_gold']})")
    show("false positives", sorted(rows_fp, key=lambda r: (r["touches_gold"], r["pecha_id"])),
         lambda r: f"iou {r['best_iou']} shaped={int(r['heading_shaped'])}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("missed", rows_missed), ("false_positives", rows_fp), ("per_book", per_book)):
        if rows:
            with (args.out_dir / f"chapter_{args.tag}_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"\nwrote CSVs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
