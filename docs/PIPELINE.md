# Pipeline and decisions

Run from the repo root. Every step reads the previous step's CSV and writes a new one, so the
cloned `.opf` files are never changed. Book texts are not in this repo. The audit, clean, split and
build steps need them, and `data/tsawa_audit.csv` has to point at your copy (see the README).

## 1. Audit

`src/audit_chapter.py` reads the raw `Chapter.yml` of every book and reports numbers only
(`data/chapter_audit.txt`). 394 books have a Chapter layer (159 old batch, 235 new batch) with
3,252 spans. Most books have very few spans: 216 of 393 have one or two.

The tsawa boundary checker reports about 72% dirty ends on the raw data. That is an artifact of
the shad convention, because chapter spans stop before the shad and the checker expects it inside.
Only about 8% of ends and 3.6% of starts (old batch) are really cut inside a syllable.

## 2. Cleaning (`src/clean_chapter_spans.py`)

Outputs `data/chapter_spans_clean.csv` (one row per span, with a `dropped` flag) and
`data/chapter_book_verdicts.csv` (one row per book).

- **Edge snapping.** A start or end cut inside a syllable is moved to the nearest boundary within 3
  characters, and spans stop before the trailing shad. 316 spans were snapped.
- **Merging.** Only spans separated by shad, tsheg, spaces or tabs on the same line are merged
  (never across a newline), so each line stays its own heading. The first version merged across
  newlines: 391 merges, 3,252 spans down to 2,861, with chains of 55 and 67 spans that fused
  separate headings. It was dropped. The final rule makes 5 merges (3,252 to 3,247).
- **Book exclusions (16).** One book has unfixable edges (P000199). Fifteen old-batch books have
  offsets that drift partway through: the first few spans are correct and every span after some
  point is cut mid-sentence. A constant shift does not recover them, so the books are left out. In
  the five largest cases checked in a browser page, only 27 of 131 spans were correct. The
  drift check is a heading-shape heuristic (fewer than 50% heading-shaped spans with 3 or more
  spans).
- **Short spans are kept.** 306 raw spans are under 15 characters, such as the numbered chapter
  lines `ལེའུ་བརྒྱད་པ`, plus a few 2-character stubs such as `༼ཇ`.

After cleaning there are 3,042 active spans in 378 books.

## 3. Split (`src/prepare_chapter_split.py`)

The first split balanced only window count (304 / 37 / 37 books, 2,292 / 477 / 273 spans) and was
replaced before any model was trained. The final split is a book-level greedy multi-objective
assignment: books largest first, then single-book moves and swaps, best of 14 seeds. It
minimises the deviation of four quantities from an 83/8.5/8.5 target across train, validation
and test: window share, chapter spans per window, share of short spans (under 30 characters), and
old-batch window share. The 97 books that are already in the tsawa split keep their split, so
validation and test books stay frozen across layers. The only hard constraints are one book in one
split and at least 30 books in validation and in test. Books that share text are not grouped.

| Split | Books | Old / new | Windows | Spans | Spans per window | Short spans | Old-batch windows |
|---|---|---|---|---|---|---|---|
| train | 308 | 114 / 194 | 9,165 (82.8%) | 2,509 | 0.274 | 31.8% | 49.3% |
| validation | 34 | 13 / 21 | 965 (8.7%) | 275 | 0.285 | 34.5% | 49.2% |
| test | 36 | 16 / 20 | 940 (8.5%) | 258 | 0.274 | 32.2% | 49.3% |

One consequence: I2EAAF38A, a book with 130 numbered outline items, was placed in validation by the
tsawa split, not by this one.

**Leakage.** Text shared between books is not treated as leakage for this project, so it is
reported and not constrained (`src/check_chapter_leakage.py`). Long spans (40+ characters) that also
appear word for word in a train book: 12 of 138 in validation (8.7%, 14 of 239 or 5.9% before the
rebuild) and 21 of 152 in test (13.8%, 11 of 116 or 9.5% before). The test leaks are concentrated in a
few books (I52248444 has 6). Exact-title matches including short spans are 8.7% (validation) and
15.1% (test), mostly stock headings such as `དཀར་ཆག` and publisher's notes. 439 book pairs share at
least one long span, 185 of them in different splits.

## 4. Dataset build (`src/build_chapter_dataset.py`)

Uses the tsawa builder (`src/build_tsawa_dataset.py`, functions `label_tokens`, `sliding_windows`,
`pack_window`) with the labels `O`, `B-CHAPTER`, `I-CHAPTER`. Windows are 8,192 tokens with stride
5,120 (mmBERT tokenizer), labels use the token-start rule, and CLS, SEP and padding are -100.
Counts are in `data/chapter_dataset_stats.json`. `src/push_chapter_dataset.py` uploads the result to
Hugging Face.

Train tokens are 74,840,009 `O`, 3,688 `B-CHAPTER` and 129,338 `I-CHAPTER` (windows overlap, so a
token is counted once per window it appears in).

## 5. Training

```
python src/train.py --label-name CHAPTER --scheme bio --weight-scheme sqrt_inv --skip-test \
    --epochs 8 --evals-per-epoch 4 --patience 3 --batch-size 8 --grad-accum 1 \
    --output-dir runs/chapter
```

The weights are `sqrt(n_O / n_class)` with `O` fixed at 1.0, which gives `B` 142.5 and `I` 24.1.
Plain inverse frequency would give about 20,300 and 580, and it was never run for Chapter, so the
choice of square root was not compared. The other settings are the script defaults (learning rate
1e-5, weight decay 0.01, gradient clip 0.3, warmup 6% which is 549 steps, break penalty 5.0 for the
trainer's own validation metric, seed 42, bf16). Gradient checkpointing stayed on, while the tsawa
and sabche runs turned it off (`--no-grad-checkpointing`). That only changes memory and speed.

The run was done on a rented GPU machine, in this order:

1. A smoke run to check the setup.
2. A 15-epoch run, stopped after about 54 steps because it would have taken about 8 hours.
3. The final 8-epoch run, saved as the best checkpoint by validation IoU 0.5 F1 and uploaded to
   Hugging Face. No other settings were tried: no second seed, learning rate or CRF.

The trainer log and results file are not in this repo (they were on the training machine, and the
log file came out empty), so the GPU model, the training time and the best epoch are not
recorded here. The training arguments are stored with the
model on Hugging Face (`training_args.bin`).

## 6. Evaluation

```
python src/eval_viterbi_iou.py --model Yontenn/mmbert-chapter-v1 \
    --dataset Yontenn/formatting-chapter-v1 --split validation --break-penalty 4 \
    --dump-spans results/mmbert-chapter-v1/dumps/val_spans.jsonl --out results/val_eval.json
python src/mmbert_dump_to_chars.py --dump results/mmbert-chapter-v1/dumps/val_spans.jsonl \
    --texts-dir data/raw_opf --out results/mmbert-chapter-v1/val --key viterbi_bp4.0_tok
python src/score_spans.py --split val --model mmbert-chapter-v1=results/mmbert-chapter-v1/val/spans
```

Use `--split test` and the test dump for the test numbers. The first command needs a GPU and was run
on the training machine. The second and third need only the tokenizer and the offsets. The dumps
are in `results/mmbert-chapter-v1/dumps/`, so the last two steps can be repeated here.

`src/eval_viterbi_iou.py` scores per window, which counts a span twice when it falls in two
overlapping windows. Its raw JSON output is kept in `results/mmbert-chapter-v1/window_level_*.json`
for the record, but the scores in the README come from the whole-book scorer.

`src/mmbert_predict.py` runs a trained model straight over a book's text, with no tokenized dataset, and
writes the same per-book offset files that `src/score_spans.py` reads:

```
python src/mmbert_predict.py --model Yontenn/mmbert-chapter-v1 --texts-dir data/raw_opf \
    --books <book ids> --out results/mmbert-chapter-v1/test --break-penalty 4.0
```

Use a GPU for whole splits, since a window of 8,192 tokens takes tens of seconds on a CPU. It reproduces
the saved predictions closely but not always exactly. On three Chapter test books it matched two exactly, and on the third it left out one 6-character span whose scores are borderline, because a CPU run and the original GPU run round slightly differently. The reported scores come from the GPU dump route
described above.

`src/analyze_chapter_errors.py` is a separate error-analysis tool. It converts the dump to
characters in memory and merges overlapping spans from different windows into one, where the
shared converter keeps a span unless it overlaps an earlier one by IoU 0.5. That is why it gives
test F1 0.836 against 0.829 for the shared scorer. On validation the two agree (0.620). Use the
shared scorer for any number you quote. The error analysis was run on validation only.

## 7. What the model gets wrong (validation)

- **One book decides the validation score.** I2EAAF38A has 130 of the 275 validation spans (47%).
  They are numbered outline items such as `༡ བཤད་བྱའི་ཡན་ལག` and `༢ གདམས་པ་དངོས`, and the model finds
  5 of them. Those 125 misses are 94% of the 133 misses on validation. The book also has 126 sabche
  spans. Without it the model finds 137 of 145 gold spans (recall 0.945, F1 0.859).
- Train has only 98 spans that start with a digit or bracket (I44B84506 has 30, I058DD999 has 15),
  validation has 94 (90 of them in that book) and test has 13.
- Miss rate by kind, validation: spans that start with a digit or bracket 90 of 94 (96%); under 15
  characters 17 of 23; 15 to 29 characters 51 of 72; 30 to 59 characters 56 of 93; 60 or more 9 of
  87. The short-span misses are mostly the same outline book, so short length is not an independent
  cause here.
- When a heading is found the boundaries are good: F1 is the same from IoU 0.5 to 0.9 and drops a
  little at exactly 1.0.
- Viterbi decoding raises precision and leaves recall alone (per window: test precision 0.458 to
  0.798, validation 0.570 to 0.773, recall unchanged).
- **False positives on validation: 41.** None touches a gold span. 34 look heading-shaped, and 20 are
  in one compilation book (I4FD99A33). By reading the text and the other layers (not checked against
  annotation rules), 24 are titles of works inside compilations (`༄༅། །title` after a colophon), 18 of
  which are annotated as book titles, and 10 are table-of-contents or front-matter lines.
  The lists are in `results/error_analysis/`.
- **A convention conflict in the labels.** Titles of works inside compilations are annotated as
  chapter in some books and as book title in others. Train has 879 chapter spans that open with
  `༄༅`, 248 of them right after a colophon, and none of those 248 is also a book title. Train has 241
  book-title lines, 10 of which are also chapter, and only 13 of the 2,509 train chapter spans
  overlap a book title. So the same kind of line can be a false positive in one book and a gold span
  in another.

## 8. Known limits

- Annotation is sparse. 216 of 393 books have one or two spans and were never audited.
- About 1.1% of new-batch chapter spans overlap a book-title span, and this was not resolved.
- The test set is small (258 spans in 36 books), so scores are noisy.
- The validation score depends on one book, so it says little about the model on its own.
- The training and evaluation scripts print reference lines about the tsawa runs. They do not apply
  to chapter.
