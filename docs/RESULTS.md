# Results

## How it is scored

A predicted span is correct when its character IoU with a gold span is at least 0.5, matched
greedily one to one. Scores are whole-book: each book is decoded once with Viterbi (break penalty
4.0), overlapping windows are de-duplicated, and totals are micro-averaged over books. This is the
same converter (`src/mmbert_dump_to_chars.py`) and scorer (`src/score_spans.py`) as in the tsawa and
sabche repos.

## Test split (36 books, 258 gold spans)

| | F1 | Precision | Recall |
|---|---|---|---|
| All 36 books | 0.829 | 0.768 | 0.899 |
| Old batch (16 books) | 0.863 | 0.853 | 0.873 |
| New batch (20 books) | 0.800 | 0.705 | 0.924 |
| Median book | 0.928 | | |

The model finds 232 of the 258 gold spans and predicts 302 spans. Reproduce without a GPU:

```
python src/score_spans.py --split test --per-book \
  --model mmbert-chapter-v1=results/mmbert-chapter-v1/test/spans
```

### Per book (test)

| Book | Batch | Gold | F1 | False positives |
|---|---|---|---|---|
| IC6F06BCD | new | 10 | 0.486 | 18 |
| P000275 | old | 54 | 0.847 | 14 |
| I9B6A4525 | new | 10 | 0.621 | 10 |
| I7989272E | new | 4 | 0.615 | 5 |
| I0FCFA88F | new | 7 | 0.778 | 4 |
| IC05A6BE0 | new | 7 | 0.778 | 4 |
| IE5895799 | new | 25 | 0.923 | 3 |
| I99DF06AA | new | 1 | 0.500 | 2 |
| IAAADCAA0 | new | 9 | 0.778 | 2 |
| P000110 | old | 8 | 0.462 | 2 |
| I4DFE9067 | new | 2 | 0.800 | 1 |
| I575514A8 | new | 7 | 0.933 | 1 |
| I9D9C7AC9 | new | 1 | 0.667 | 1 |
| P000067 | old | 7 | 0.769 | 1 |
| P000118 | old | 4 | 0.889 | 1 |
| P000245 | old | 24 | 0.980 | 1 |
| I319DAFF7 | new | 1 | 1.000 | 0 |
| I3F4A91F5 | new | 11 | 1.000 | 0 |
| I45222122 | new | 1 | 1.000 | 0 |
| I52248444 | new | 15 | 0.966 | 0 |
| I5F5D9F5A | new | 8 | 0.857 | 0 |
| I62D7430C | new | 1 | 1.000 | 0 |
| I9AEEF96A | new | 1 | 1.000 | 0 |
| ICDC84458 | new | 9 | 0.941 | 0 |
| IE9C4806D | new | 2 | 0.667 | 0 |
| P000013 | old | 2 | 1.000 | 0 |
| P000023 | old | 1 | 1.000 | 0 |
| P000027 | old | 2 | 1.000 | 0 |
| P000056 | old | 2 | 0.667 | 0 |
| P000079 | old | 13 | 0.818 | 0 |
| P000109 | old | 1 | 1.000 | 0 |
| P000115 | old | 3 | 1.000 | 0 |
| P000134 | old | 1 | 1.000 | 0 |
| P000138 | old | 2 | 1.000 | 0 |
| P000172 | old | 1 | 1.000 | 0 |
| P000223 | old | 1 | 1.000 | 0 |

Most books have only a handful of gold spans, so single books move the score: P000275 (54 gold
spans) has 14 false positives, and IC6F06BCD, IE5895799 and I9B6A4525 (10 to 25 gold spans) have 18, 3
and 10.

## Validation (34 books, 275 gold spans)

| | F1 | Precision | Recall |
|---|---|---|---|
| All 34 books | 0.620 | 0.776 | 0.516 |
| Old batch (13 books) | 0.875 | 0.854 | 0.897 |
| New batch (21 books) | 0.566 | 0.754 | 0.453 |
| Without I2EAAF38A | 0.859 | 0.787 | 0.945 |

I2EAAF38A is one outline-style book with 130 of the 275 validation spans (47%), and the model finds
5 of them. Without it the model finds 137 of 145 gold spans. The per-book validation table is in
`results/error_analysis/val_per_book.csv`, the 290 missed spans in `val_missed.csv`, and the 41
false positives in `val_false_positives.csv`.

## Files

- `results/mmbert-chapter-v1/{test,val}/spans/<book>.json`: predicted spans as inclusive character
  offsets, ready for `src/score_spans.py`
- `results/mmbert-chapter-v1/dumps/`: the per-window token-span dumps they were made from
- `results/mmbert-chapter-v1/window_level_*.json`: the per-window scores from `eval_viterbi_iou.py`,
  kept for the record (they count a span twice when it falls in two overlapping windows)
- `results/mmbert-chapter-v1/hf_model_config.json`: the model config from Hugging Face
