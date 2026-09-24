#!/usr/bin/env python3
"""
Frozen book-level split for the Chapter dataset, stratified.

Usage:
    python src/prepare_chapter_split.py [--keep-v3] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from build_tsawa_dataset import sliding_windows  # noqa: E402

AUDIT = ROOT / "data/tsawa_audit.csv"
CLEAN = ROOT / "data/chapter_spans_clean.csv"
VERDICTS = ROOT / "data/chapter_book_verdicts.csv"
SPLIT_V3 = ROOT / "data/tsawa_split_frozen.csv"
OUT_SPLIT = ROOT / "data/split_frozen.csv"
OUT_REPORT = ROOT / "results/chapter_split_report.json"
TOKENIZER = "jhu-clsp/mmBERT-base"
CONTENT_LEN = 8190  # 8192 - CLS - SEP
STRIDE = 5120
SHORT_MAX = 29      # a span under 30 characters counts as short
MIN_DOCS = 30
TARGETS = {"train": 0.83, "val": 0.085, "test": 0.085}
SEEDS = [0, 1, 2, 3, 7, 11, 13, 17, 19, 42, 99, 123, 256, 2026]
WEIGHTS = {"windows": 3.0, "spw": 2.0, "short": 1.0, "batch": 1.0}
SPLITS = ("train", "val", "test")


def split_stats(rows, assign):
    st = {s: {"docs": 0, "win": 0, "spans": 0, "short": 0, "old": 0} for s in SPLITS}
    for r in rows:
        s = st[assign[r["id"]]]
        s["docs"] += 1
        s["win"] += r["win"]
        s["spans"] += r["spans"]
        s["short"] += r["short"]
        s["old"] += r["win"] if r["batch"] == "old" else 0
    return st


def score(st, g):
    tot_w = sum(x["win"] for x in st.values()) or 1
    sc = 0.0
    for sp in SPLITS:
        x = st[sp]
        w = max(x["win"], 1)
        sc += WEIGHTS["windows"] * ((x["win"] / tot_w - TARGETS[sp]) / TARGETS[sp]) ** 2
        sc += WEIGHTS["spw"] * ((x["spans"] / w - g["spw"]) / g["spw"]) ** 2
        sc += WEIGHTS["short"] * ((x["short"] / max(x["spans"], 1) - g["short"]) / g["short"]) ** 2
        sc += WEIGHTS["batch"] * ((x["old"] / w - g["old"]) / g["old"]) ** 2
    return sc


def assign_books(rows, fixed, g, seed):
    rng = random.Random(seed)
    assign = dict(fixed)
    free = [r for r in rows if r["id"] not in fixed]
    free.sort(key=lambda r: (-r["win"], rng.random()))
    rows_placed = [r for r in rows if r["id"] in fixed]

    def cur_score(extra):
        sub = rows_placed + extra
        return score(split_stats(sub, assign), g)

    placed = list(rows_placed)
    for r in free:
        best, best_sc = None, None
        for sp in SPLITS:
            assign[r["id"]] = sp
            sc = score(split_stats(placed + [r], assign), g)
            if best_sc is None or sc < best_sc:
                best, best_sc = sp, sc
        assign[r["id"]] = best
        placed.append(r)

    def ok(a):
        st = split_stats(rows, a)
        return st["val"]["docs"] >= MIN_DOCS and st["test"]["docs"] >= MIN_DOCS

    cur = score(split_stats(rows, assign), g)
    movable = [r["id"] for r in free]
    for _ in range(6):
        improved = False
        for p in movable:  # single-book moves
            for sp in SPLITS:
                if sp == assign[p]:
                    continue
                old = assign[p]
                assign[p] = sp
                sc = score(split_stats(rows, assign), g)
                if sc + 1e-12 < cur and ok(assign):
                    cur, improved = sc, True
                else:
                    assign[p] = old
        for i, p in enumerate(movable):  # pairwise swaps
            for q in movable[i + 1:]:
                if assign[p] == assign[q]:
                    continue
                assign[p], assign[q] = assign[q], assign[p]
                sc = score(split_stats(rows, assign), g)
                if sc + 1e-12 < cur and ok(assign):
                    cur, improved = sc, True
                else:
                    assign[p], assign[q] = assign[q], assign[p]
        if not improved:
            break
    return assign, cur


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep-v3", action="store_true",
                    help="books in tsawa split_v3 keep their split")
    ap.add_argument("--out", type=Path, default=OUT_SPLIT)
    args = ap.parse_args(argv)

    audit = {r["pecha_id"]: r for r in csv.DictReader(AUDIT.open(encoding="utf-8"))}
    verdict = {r["pecha_id"]: r for r in csv.DictReader(VERDICTS.open(encoding="utf-8"))}
    v3 = {r["pecha_id"]: r["split"] for r in csv.DictReader(
        l for l in SPLIT_V3.open(encoding="utf-8") if not l.startswith("#"))}
    excluded = {p for p, v in verdict.items() if v["verdict"] == "exclude"}

    lens = defaultdict(list)
    for r in csv.DictReader(CLEAN.open(encoding="utf-8")):
        if r["dropped"] != "True" and r["pecha_id"] not in excluded:
            lens[r["pecha_id"]].append(int(r["end"]) - int(r["start"]))
    books = sorted(lens)
    print(f"books kept {len(books)}  excluded {len(excluded)}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    tok.model_max_length = int(1e12)
    rows, n_tok = [], {}
    for p in books:
        t = Path(audit[p]["base_path"]).read_text(encoding="utf-8")
        n_tok[p] = len(tok(t, add_special_tokens=False)["input_ids"])
        rows.append({"id": p, "batch": "old" if p.startswith("P") else "new",
                     "win": len(sliding_windows(n_tok[p], CONTENT_LEN, STRIDE)),
                     "spans": len(lens[p]), "short": sum(L <= SHORT_MAX for L in lens[p])})
    tw = sum(r["win"] for r in rows)
    g = {"spw": sum(r["spans"] for r in rows) / tw,
         "short": sum(r["short"] for r in rows) / sum(r["spans"] for r in rows),
         "old": sum(r["win"] for r in rows if r["batch"] == "old") / tw}
    fixed = {p: v3[p] for p in books if p in v3} if args.keep_v3 else {}
    print(f"global: spans/window {g['spw']:.3f}  short {100 * g['short']:.1f}%  old windows {100 * g['old']:.1f}%")
    print(f"fixed from split_v3: {len(fixed)}")

    best = None
    for seed in SEEDS:
        a, sc = assign_books(rows, fixed, g, seed)
        print(f"  seed {seed:4}: score {sc:.5f}")
        if best is None or sc < best[1]:
            best = (a, sc, seed)
    assign, sc, seed = best
    st = split_stats(rows, assign)

    print(f"\nbest seed {seed}  score {sc:.5f}")
    print(f"{'split':6}{'books':>6}{'old/new':>9}{'windows':>9}{'win%':>7}{'spans':>7}{'spans/win':>10}{'short%':>8}{'old-win%':>9}")
    stats = {}
    for sp in SPLITS:
        x = st[sp]
        n_old = sum(1 for r in rows if assign[r["id"]] == sp and r["batch"] == "old")
        stats[sp] = {"books": x["docs"], "old_books": n_old, "new_books": x["docs"] - n_old,
                     "windows": x["win"], "pct_windows": round(100 * x["win"] / tw, 2),
                     "spans": x["spans"], "spans_per_window": round(x["spans"] / x["win"], 3),
                     "pct_short": round(100 * x["short"] / max(x["spans"], 1), 1),
                     "pct_old_windows": round(100 * x["old"] / x["win"], 1),
                     "from_split_v3": sum(1 for r in rows if assign[r["id"]] == sp and r["id"] in fixed)}
        s = stats[sp]
        print(f"{sp:6}{s['books']:6}{s['old_books']:>4}/{s['new_books']:<4}{s['windows']:9}{s['pct_windows']:7.1f}"
              f"{s['spans']:7}{s['spans_per_window']:10.3f}{s['pct_short']:8.1f}{s['pct_old_windows']:9.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        fh.write(
            "# chapter layer-detection document split - FROZEN\n"
            f"# generated: {date.today().isoformat()}\n"
            "# generator: src/prepare_chapter_split.py (greedy multi-objective)\n"
            f"# seed: {seed}\n"
            f"# targets: {TARGETS['train']:.0%}/{TARGETS['val']:.1%}/{TARGETS['test']:.1%} by WINDOW count (8192 / 5120)\n"
            "# stratified on: batch, n_windows, chapter spans per window, pct_short (<30 chars)\n"
            f"# split_v3 assignments kept: {args.keep_v3}\n"
            "# constraint: one book, one split; no grouping of books that share text\n"
            "# span source: data/chapter_spans_clean.csv (dropped=False)\n"
            "# TEST SPLIT IS FROZEN: evaluate test once at the end, never for tuning.\n")
        w = csv.DictWriter(fh, fieldnames=["pecha_id", "split", "n_tokens", "n_windows", "n_chapter_spans"])
        w.writeheader()
        for r in rows:
            w.writerow({"pecha_id": r["id"], "split": assign[r["id"]], "n_tokens": n_tok[r["id"]],
                        "n_windows": r["win"], "n_chapter_spans": r["spans"]})
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps({"seed": seed, "score": sc, "global": g, "stats": stats,
                                      "keep_v3": args.keep_v3, "excluded": sorted(excluded)}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
