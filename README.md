# chapter-layer-detection

Finding the **chapter** layer (ལེའུ་, the heading line that opens a division of a book) in
Tibetan texts with a fine-tuned `jhu-clsp/mmBERT-base` token classifier. This repo has the whole
pipeline: how the labels were cleaned, how the split was built, how the model was trained and
how it is scored.

- Dataset: [Yontenn/formatting-chapter-v1](https://huggingface.co/datasets/Yontenn/formatting-chapter-v1)
- Model: [Yontenn/mmbert-chapter-v1](https://huggingface.co/Yontenn/mmbert-chapter-v1)

## 1. What chapter is

A chapter span is the heading line that opens a division of a book. It names the division and is
not the text under it. Only 5.6% of spans are numbered chapter lines such as `ལེའུ་བཅུ་བཞི་པ`.
The rest are the titles of separate works inside a collected volume (usually opening with
`༄༅། །`, often ending `ཞེས་བྱ་བ་བཞུགས་སོ`), plain section titles inside a longer work, and front
matter such as `དཀར་ཆག` (table of contents) or `དཔེ་སྐྲུན་གསལ་བཤད` (publisher's note). Spans are
short (median 43 characters), almost always start at the beginning of a line, and are sparse:
more than half of the books have only one or two. The annotations come from OpenPecha `.opf`
books (`Chapter.yml`).

## 2. Data preparation

Source: 394 books with a Chapter layer (159 old batch, 235 new batch), 3,252 spans. Books with no
Chapter layer are not in the dataset. The raw annotations are never edited; every fix is written
to a sidecar CSV in `data/`. More detail is in `docs/PIPELINE.md`.

| Step | What was done | Effect |
|---|---|---|
| Edge snapping | A start or end cut inside a syllable is moved to the nearest boundary within 3 characters. Spans stop before the trailing shad. | 316 spans snapped. The tsawa boundary checker reports about 72% dirty ends on the raw data, which is an artifact of the shad convention. Only about 8% of ends and 3.6% of starts (old batch) were really mid-syllable. |
| Merging | Only spans separated by shad, tsheg or spaces on the same line are merged. Each line stays its own heading. The first version also merged across newlines and fused separate headings into chains of 55 and 67 spans, so it was dropped. | 5 merges (3,252 to 3,247 spans). The first version made 391. |
| Book exclusions | One book has unfixable edges. Fifteen old-batch books have offsets that drift partway through, and a constant shift does not recover them. | 16 books left out. In the five largest drift books only 27 of 131 spans were correct. |
| Short spans | Kept. 306 raw spans are under 15 characters, such as `ལེའུ་བརྒྱད་པ`. | 0 dropped. |
| Split | Book-level, stratified on window share, spans per window, share of short spans and old-batch share, 83/8.5/8.5 by windows. The 97 books already in the tsawa split keep their split. Books that share text are not grouped. | The first split (by window count only) was replaced: it had 2,292 / 477 / 273 spans in train / val / test. The final split has 2,509 / 275 / 258. |

Leakage was measured and not constrained. Long spans (40+ characters) that also appear word for
word in a train book: 12 of 138 in validation (8.7%) and 21 of 152 in test (13.8%), concentrated
in a few books. Exact-title matches are mostly stock headings (`དཀར་ཆག`, publisher's notes). 439
book pairs share at least one long span, 185 of them in different splits.

Known limits: 216 of the 393 books have one or two spans and may be under-annotated (not
audited), and about 1.1% of new-batch chapter spans overlap a book-title span. The validation
split is unusual: one book holds 47% of its spans (see section 5).

## 3. Dataset

Hugging Face: [Yontenn/formatting-chapter-v1](https://huggingface.co/datasets/Yontenn/formatting-chapter-v1)
(card in `data/dataset_card.md`). Windows of 8,192 tokens with stride 5,120 (mmBERT tokenizer),
BIO labels (`O`, `B-CHAPTER`, `I-CHAPTER`). The frozen split is `data/split_frozen.csv`.

| Split | Books (old / new) | Windows | Chapter spans |
|---|---|---|---|
| train | 308 (114 / 194) | 9,165 | 2,509 |
| validation | 34 (13 / 21) | 965 | 275 |
| test | 36 (16 / 20) | 940 | 258 |

The test split is frozen. Train label counts are 74,840,009 `O`, 3,688 `B` and 129,338 `I` tokens
(windows overlap, so tokens are counted once per window). Chapter is about 0.18% of tokens,
much sparser than the other layers.

## 4. Training

`src/train.py`, giving [Yontenn/mmbert-chapter-v1](https://huggingface.co/Yontenn/mmbert-chapter-v1).
This is the same script as in the tsawa and sabche repos, run with `--label-name CHAPTER`.

| | |
|---|---|
| Base model | `jhu-clsp/mmBERT-base`, token classification, 3 labels |
| Learning rate / batch size | 1e-5 / 8 |
| Epochs | 8, early stopping with patience 3 epochs, evaluated 4 times per epoch |
| Class weights | square-root inverse frequency: `O` 1.0, `B` 142.5, `I` 24.1 |
| Other | weight decay 0.01, gradient clip 0.3, warmup 549 steps, bf16, gradient checkpointing, seed 42 |

The class weights are square-root inverse frequency because plain inverse frequency would give
`B` about 20,300 and `I` about 580. Plain inverse frequency was never tried for Chapter.
A 15-epoch run was started and stopped after about 54 steps because it would have taken about 8
hours, so the final run is the same setting with 8 epochs. The test split was never scored
during training. The command is in `docs/PIPELINE.md`.

## 5. Evaluation

Predictions are decoded with Viterbi (break penalty 4.0) into character spans. A predicted span
is correct when its IoU with a gold span is at least 0.5, matched greedily one to one. Each book
is decoded once, with duplicates from the overlapping windows removed, and scores are
micro-averaged over books. This is the same converter and scorer as in the tsawa and sabche
repos.

| | F1 | Precision | Recall |
|---|---|---|---|
| **Test, 36 books, 258 gold spans** | **0.829** | 0.768 | 0.899 |
| Test, old batch (16 books) | 0.863 | 0.853 | 0.873 |
| Test, new batch (20 books) | 0.800 | 0.705 | 0.924 |
| Test, median book | 0.928 | | |
| Validation, 34 books, 275 gold spans | 0.620 | 0.776 | 0.516 |
| Validation, old batch (13 books) | 0.875 | 0.854 | 0.897 |
| Validation, new batch (21 books) | 0.566 | 0.754 | 0.453 |

The model predicts 302 spans for the 258 test gold spans. You can check the test and validation
numbers without a GPU:

```
python src/score_spans.py --split test --per-book \
  --model mmbert-chapter-v1=results/mmbert-chapter-v1/test/spans
```

The validation score is low because of one book. I2EAAF38A holds 130 of the 275 validation
spans (47%), numbered outline items such as `༡ བཤད་བྱའི་ཡན་ལག`, and the model finds 5 of them.
Without that book, validation is F1 0.859 (precision 0.787, recall 0.945). Train has only 98
spans that start with a digit or bracket, against 94 in validation (90 of them in that book), so
this is a style the model has barely seen. The test set is small, with 258 spans in 36 books, so
scores are noisy.

The test split was scored after training, with the break penalty of 4.0 chosen beforehand and
never changed. Model inference on the test set ran twice with the same final model, and the
predictions were rescored once without a model. Nothing was changed afterwards.

## 6. Summary

| Model | Score | Notes |
|---|---|---|
| Chapter model, validation, whole books | 0.620 F1 | 0.859 without the outline book |
| Chapter model, test, whole books | **0.829 F1** | 36 books, P 0.768 / R 0.899 |

## Layout

```
data/     dataset card, frozen split, cleaned spans, book verdicts, gold spans, the tsawa
          audit and split that the chapter split builds on
src/      audit, clean, split, build dataset, train, evaluate
docs/     PIPELINE.md (why each decision was made), RESULTS.md (all numbers)
results/  mmBERT predictions as character offsets, raw dumps, validation error analysis
```

Book texts are not included. They come from OpenPecha. `data/tsawa_audit.csv` records where each
book's text was on the original machine (`base_path`), so point those paths at your own copy of
the texts before running the audit, clean, split or build scripts.
