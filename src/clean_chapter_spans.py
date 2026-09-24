#!/usr/bin/env python3
"""
Build the cleaned Chapter span sidecar. Does not rewrite any Chapter.yml.

Usage:
    python src/clean_chapter_spans.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
from check_boundary_snapping import BOUNDARY_CHARS  # noqa: E402

AUDIT = ROOT / "data/tsawa_audit.csv"
OUT = ROOT / "data/chapter_spans_clean.csv"
OUT_BOOKS = ROOT / "data/chapter_book_verdicts.csv"

MAX_WALK = 3
EXCLUDE_DIRTY = 0.30
MIN_SPANS_SHAPE = 3
EXCLUDE_SHAPE_BELOW = 0.50
PRE = re.compile(r"(^|\n)[༈༄༅།\s\d༠-༩\(\)\{\}\[\]༼༽\.]*$")
POST = re.compile(r"^[།༎་\s\xa0\]\)\}༽]*(\n|$)")
GAP_CHARS = set("།༎་ \t\xa0")  # no newline: each line is its own heading


def load_chapter(opf: Path) -> list[tuple[str, int, int]]:
    p = opf / "layers/v001/Chapter.yml"
    if not p.is_file():
        return []
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    anns = d.get("annotations") or {}
    items = anns.items() if isinstance(anns, dict) else enumerate(anns)
    out = []
    for aid, a in items:
        sp = (a or {}).get("span") or {}
        s, e = sp.get("start"), sp.get("end")
        if s is None or e is None or int(e) <= int(s):
            continue
        out.append((str(aid), int(s), int(e)))
    return sorted(out, key=lambda x: (x[1], x[2]))


def is_b(ch: str) -> bool:
    return ch in BOUNDARY_CHARS


def mid_start(t, s):
    return 0 < s < len(t) and not is_b(t[s - 1]) and not is_b(t[s])


def mid_end(t, e):
    return 0 < e < len(t) and not is_b(t[e - 1]) and not is_b(t[e])


def snap(text: str, s: int, e: int) -> tuple[int, int, str]:
    n, act = len(text), []
    if mid_start(text, s):
        for k in range(1, MAX_WALK + 1):
            if s - k == 0 or is_b(text[s - k - 1]):
                s -= k
                act.append(f"start-{k}")
                break
        else:
            act.append("start_unfixed")
    if mid_end(text, e):
        for k in range(1, MAX_WALK + 1):
            if e + k >= n or is_b(text[e + k]):
                e += k
                act.append(f"end+{k}")
                break
        else:
            act.append("end_unfixed")
    return s, e, "|".join(act)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows, verdicts, tot = [], [], Counter()
    for r in csv.DictReader(AUDIT.open(encoding="utf-8")):
        pid = r["pecha_id"]
        raw = load_chapter(Path(r["opf_root"]))
        if not raw:
            continue
        batch = "old" if pid.startswith("P") else "new"
        text = Path(r["base_path"]).read_text(encoding="utf-8")
        sn = []
        for aid, s0, e0 in raw:
            s, e, act = snap(text, s0, e0)
            sn.append({"ann_id": aid, "raw_start": s0, "raw_end": e0, "start": s,
                       "end": e, "action": act or "keep", "n_merged": 1})
        dirty = sum(("unfixed" in x["action"]) for x in sn) / len(sn)
        for a, b in zip(sn, sn[1:]):
            if b["start"] < a["end"]:
                a["end"] = max(b["start"], a["start"] + 1)
                a["action"] += "|trim_overlap"
                tot["trimmed"] += 1
        merged = [sn[0]]
        for b in sn[1:]:
            a = merged[-1]
            if set(text[a["end"]:b["start"]]) <= GAP_CHARS:
                a["end"] = b["end"]
                a["n_merged"] += b["n_merged"]
                a["action"] += "|merge"
                tot["merged_away"] += 1
            else:
                merged.append(b)
        shaped = sum(bool(PRE.search(text[max(0, m["start"] - 6):m["start"]])
                          and POST.match(text[m["end"]:m["end"] + 8])) for m in merged)
        shape_rate = shaped / len(merged)
        bad_shape = len(merged) >= MIN_SPANS_SHAPE and shape_rate < EXCLUDE_SHAPE_BELOW
        verdict = "exclude" if dirty > EXCLUDE_DIRTY or bad_shape else "keep"
        reason = ("edges_unfixable" if dirty > EXCLUDE_DIRTY
                  else "offsets_drift_not_heading_shaped" if bad_shape else "")
        for m in merged:
            rows.append({"pecha_id": pid, "batch": batch, **m,
                         "dropped": verdict == "exclude",
                         "reason": reason})
        tot["raw"] += len(raw)
        tot["final"] += len(merged)
        tot["snapped"] += sum(("start-" in x["action"] or "end+" in x["action"]) for x in sn)
        tot["unfixed"] += sum("unfixed" in x["action"] for x in sn)
        verdicts.append({"pecha_id": pid, "batch": batch, "n_raw": len(raw),
                         "n_final": len(merged), "unfixed_rate": round(dirty, 3),
                         "shape_rate": round(shape_rate, 3),
                         "verdict": verdict})
    for path, data in ((OUT, rows), (OUT_BOOKS, verdicts)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    vc = Counter((v["batch"], v["verdict"]) for v in verdicts)
    print("books: " + "  ".join(f"{b}/{k}={n}" for (b, k), n in sorted(vc.items())))
    print(f"spans: raw {tot['raw']:,} -> final {tot['final']:,} (merged away {tot['merged_away']:,}); "
          f"snapped {tot['snapped']:,}  left-dirty {tot['unfixed']:,}  overlap-trimmed {tot['trimmed']:,}")
    print(f"wrote {OUT}\nwrote {OUT_BOOKS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
