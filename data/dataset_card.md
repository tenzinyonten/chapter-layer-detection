---
pretty_name: Formatting Chapter
language:
- bo
task_categories:
- token-classification
tags:
- tibetan
- chapter
- bio
- mmbert
- openpecha
size_categories:
- 10K<n<100K
license: other
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Chapter (ལེའུ་, chapter heading) dataset - mmBERT BIO

Binary token classification for the Chapter layer, built with the same
labeling and windowing as the tsawa and sabche datasets. Generated 2026-09-24 (split rebuilt with stratification the same day).

Split: 83/8.5/8.5 by window count, stratified on batch, spans per window and
share of short spans (`split_frozen.csv`), by book only.

| item | value |
|---|---|
| tokenizer | `jhu-clsp/mmBERT-base` (fast, offsets) |
| window / step | 8192 (8190 content + CLS/SEP) / 5120 between window starts |
| labels | `O`=0, `B-CHAPTER`=1, `I-CHAPTER`=2; CLS/SEP/pad = -100 |
| label rule | token-start rule (`build_tsawa_dataset.label_tokens`) |
| columns | input_ids, attention_mask, labels, token_start, token_end, char_start, char_end, pecha_id, source_batch, window_index, n_tokens_doc, coverage_pct |
| span source | `data/chapter_spans_clean.csv` (dropped=False) |
| split | `data/split_frozen.csv` (**test frozen**) |

**Geometry.** Windows hold 8,192 tokens (8,190 content tokens plus CLS and SEP) and start every
5,120 tokens, so consecutive windows overlap by 3,070 content tokens. In this card and in the code,
"stride" means the step between window starts. The Hugging Face tokenizer argument `stride` means the
overlap, so the equivalent value there is 3,070. Tokens in the overlap are labelled in both windows.

**Offsets.** Span offsets in the cleaned span CSVs and in the dataset columns `char_start` and
`char_end` use an exclusive end (`text[start:end]`). The prediction files and the scoring code in the
GitHub repository use an inclusive end (`text[start:end+1]`), so convert with `end - 1`.

## Size

| split | books | old / new | windows | Chapter spans | spans per window | short spans (<30 chars) | old-batch windows |
|---|---|---|---|---|---|---|---|
| train | 308 | 114 / 194 | 9,165 (82.8%) | 2,509 | 0.274 | 31.8% | 49.3% |
| validation | 34 | 13 / 21 | 965 (8.7%) | 275 | 0.285 | 34.5% | 49.2% |
| test | 36 | 16 / 20 | 940 (8.5%) | 258 | 0.274 | 32.2% | 49.3% |

Train tokens: 74.8M `O`, 3,688 `B-CHAPTER`, 129,338 `I-CHAPTER` (windows overlap,
so a token is counted once per window it appears in). Chapter is about 0.18% of
tokens, far sparser than the other layers. `train_layer.py`'s default `inv` weights
(relative to `O`) come out around 20,000 (B) and 580 (I); `sqrt_inv` gives about
142 and 24, which is the setting to start from.

## Pipeline

```bash
python src/clean_chapter_spans.py      # sidecar + book verdicts
python src/prepare_chapter_split.py --keep-v3   # frozen, stratified book-level split
python src/build_chapter_dataset.py
python src/check_chapter_leakage.py    # optional, read-only
python src/push_chapter_dataset.py --private
```

## Cleaning

Raw `Chapter.yml` offsets from 394 books with a Chapter layer (159 old, 235 new;
3,252 spans). Books with no Chapter layer are not in the dataset.

- **Edge snapping.** A start or end cut inside a syllable is walked to the nearest
  boundary within 3 characters (316 spans). Spans stop before the trailing shad.
  The Tsawa boundary checker reports about 72% dirty ends on the raw data; that
  is an artifact of the shad convention. Only about 8% of ends and 3.6% of starts
  (old batch) were actually mid-syllable.
- **Merging.** Only spans separated by shad, tsheg or spaces (never a newline) are
  merged: 5 merges. Each line is its own heading.
- **Book exclusions (16).** One book has unfixable edges. Fifteen old-batch books
  have offsets that drift partway through: the first few spans are correct and
  every span after some point is cut mid-sentence. A constant shift does not
  recover them, so the books are left out entirely. They hold 27 correct spans
  in the five largest cases checked (out of 131 spans).
- **Short spans are kept.** 306 raw spans are under 15 characters (short chapter
  numbers like `ལེའུ་བརྒྱད་པ`), plus a few 2-character stubs such as `༼ཇ`.
- **Splitting.** A greedy multi-objective assignment (books largest first, then
  single-book moves and swaps, best of 14 seeds) minimises the deviation of window
  share, Chapter spans per window, share of short spans (under 30 characters) and
  old-batch window share across train/val/test. The 97 books already in the tsawa
  split keep their split so val/test books stay frozen across layers; the rest
  are optimised. The only hard constraints are one book, one split, and at least 30
  books in val and test. Books that share text are not grouped.

## Leakage

Text shared between different books is not treated as leakage for this project,
so it is reported and not constrained. Shingle matching on spans of 40+ characters:
12 of 138 val spans (8.7%) and 21 of 152 test spans (13.8%) also appear verbatim
in a train book; the test leaks are concentrated in a few books (I52248444 has 6).
Exact-title matches that include short spans are 8.7% (val) and 15.1% (test) and
are mostly stock headings: `དཀར་ཆག` (table of contents) and publisher's notes such
as `དཔེ་སྐྲུན་སྨོན་ཚིག` and `དཔེ་སྐྲུན་གསལ་བཤད`. 439 book pairs share at least one long
span, and 185 of them sit in different splits.

## Known limitations

- Most books have very few Chapter spans (216 of 393 have one or two before
  exclusions), and some may be under-annotated. Not audited.
- About 1.1% of new-batch Chapter spans overlap a BookTitle span. Not resolved.
- The test split has 258 spans in 36 books, so scores will be noisy.
- The heading-shape check used to find drifted books is a heuristic.

## Citation

```bibtex
@misc{formatting_chapter,
  title  = {Formatting Chapter: Tibetan chapter-heading token classification data},
  author = {Yontenn},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Yontenn/formatting-chapter-v1}
}
```

## Acknowledgements

Source texts were digitized and made available by the [Buddhist Digital Resource Center (BDRC)](https://www.bdrc.io/). We gratefully acknowledge BDRC. Annotations were prepared through [OpenPecha](https://openpecha.org/) with support from the [Tsadra Foundation](https://www.tsadra.org/).
