# A/B comparison: deterministic vs multimodal blank detection

These files are the evidence behind PaperFill's core claim: that finding blanks
with a **vision model plus an anchor bridge** beats the **deterministic
underscore/gap heuristics** it replaced.

Regenerate everything here with:

```bash
python tools/ab_compare.py
```

## What the two pipelines are

- **Deterministic** (`src/paperfill/ai/preprocess.py`) finds blanks by scanning
  the text layer for underscore runs and suspicious whitespace gaps. Fast and
  free, but it only sees what the PDF's text layer spells out.
- **Multimodal** (`src/paperfill/ai/multimodal_preprocess.py`) asks a vision
  model what the answer spaces are, then bridges those back onto real text
  anchors so the coordinates stay exact.

## The two cases that separate them

The synthetic worksheet is built to contain the two failures that motivated the
rewrite:

| Case | Deterministic result | Why |
|---|---|---|
| `definition of bob -` | **Miss.** No blank found. | The answer space is bare whitespace after a dash, with no underscore run to match. |
| `single-eyed` | **False positive.** Invents a blank. | A hyphenated word with empty space under it looks like an open-response region to the gap heuristic. |

## The files

**Synthetic worksheet** — the controlled case, both failures present:

| File | What it shows |
|---|---|
| `synthetic_worksheet.pdf` | The unfilled input. |
| `synth_deterministic_filled.pdf` / `.png` | Deterministic output. Misses the dash blank, invents one on `single-eyed`. |
| `synth_multimodal_filled.pdf` / `.png` | Multimodal output from a recorded vision response. Both cases correct. |
| `synth_multimodal_LIVE_filled.pdf`, `synth_multimodal_LIVE.png` | Same, but from a live model call rather than the recorded fixture. Confirms the recorded response is representative. |

**Real worksheets** — the same comparison on documents not built to prove a point:

| File | What it shows |
|---|---|
| `vocab_dash.pdf` → `vocab_dash_filled.pdf` / `.png` | A real vocabulary sheet using the dash-blank format. |
| `blank_guide.pdf` → `blank_guide_filled.pdf` / `.png` | A real study guide with mixed blank styles. |
| `real_multimodal_filled.pdf` | Multimodal output on a scanned (image-only) worksheet, exercising the vision path end to end. |

## Reading the harness without an API key

`tools/ab_compare.py` drives the multimodal path with a **recorded** vision
response, so the resolver, geometry, and renderer all run end to end offline.
With a valid key, `multimodal_preprocess_pdf(path)` makes the same call live —
that's what the `_LIVE` files above are.

One-off runs against your own PDFs (`dle*`) are gitignored; only this curated
set is committed.
