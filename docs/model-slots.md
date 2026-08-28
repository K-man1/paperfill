# LLM model slots

Extracted from the code, not from slot labels. Every claim cites a file and line.
Nothing here was inferred; anything the code does not determine is marked
**unknown**.

Paths are relative to the repo root. Read at commit `2fa8dfb`.

---

## 1. Inventory

### 1.1 Slot definitions

`SLOTS` in [src/paperfill/data/models.py:45-73](../src/paperfill/data/models.py). Seven
slots. Each is `ModelSlot(env, label, inherits, needs_vision, note)`
([models.py:36-42](../src/paperfill/data/models.py)).

| Slot key | Env var | Inherits | `needs_vision` | Defined |
|---|---|---|---|---|
| `vision` | `VISION_MODEL` | — | True | [models.py:46-50](../src/paperfill/data/models.py) |
| `vision_pro` | `VISION_MODEL_PRO` | `vision` | True | [models.py:51-53](../src/paperfill/data/models.py) |
| `detect` | `MULTIMODAL_MODEL` | `vision` | True | [models.py:54-58](../src/paperfill/data/models.py) |
| `regions` | `REGION_MODEL` | `vision` | True | [models.py:59-62](../src/paperfill/data/models.py) |
| `text_fill` | `AI_MODEL` | — | False | [models.py:63-65](../src/paperfill/data/models.py) |
| `text_fill_pro` | `AI_MODEL_PRO` | `text_fill` | False | [models.py:66-68](../src/paperfill/data/models.py) |
| `fallback` | `OPENROUTER_MODEL` | — | False | [models.py:69-72](../src/paperfill/data/models.py) |

`needs_vision` is **advisory metadata only**. Grep confirms it is never read
outside the admin template; it does not gate any call.

### 1.2 Resolution order

`models.get(slot)` ([models.py:84-94](../src/paperfill/data/models.py)):

1. `ai_models.json` override (admin dashboard, written by `save()` at
   [models.py:107-120](../src/paperfill/data/models.py), form posted at
   [app.py:1894-1895](../src/paperfill/app.py)).
2. The slot's env var.
3. Recursive `get(inherits)`.
4. `DEFAULT_MODEL = "openai/gpt-5.5"` ([models.py:33](../src/paperfill/data/models.py)).

Read fresh on every call — no caching ([models.py:76-81](../src/paperfill/data/models.py)).

`ai_models.json` **does not exist** in this checkout, so every slot currently
resolves through step 2 or 3.

### 1.3 Provider routing

One wrapper, `FallbackClient` ([llm_client.py:238-255](../src/paperfill/ai/llm_client.py)),
built by `build_client()` ([llm_client.py:258-282](../src/paperfill/ai/llm_client.py)).

- **Primary base URL**: `AI_BASE_URL` → `OPENAI_BASE_URL` → `https://ai.hackclub.com/proxy/v1`
  ([llm_client.py:270-272](../src/paperfill/ai/llm_client.py)).
- **Primary key**: `AI_API_KEY` → `HCAI_API_KEY` → `OPENAI_API_KEY`; raises if
  none ([llm_client.py:261-267](../src/paperfill/ai/llm_client.py)).
- **Fallback base URL**: `OPENROUTER_BASE_URL` → `https://openrouter.ai/api/v1`,
  built only if `OPENROUTER_API_KEY` is set
  ([llm_client.py:274-280](../src/paperfill/ai/llm_client.py)).

The same URL selection is duplicated for the catalog fetch at
[models.py:158-161](../src/paperfill/data/models.py) (`GET {base}/models`, 900 s
per-worker cache, [models.py:130-155](../src/paperfill/data/models.py)).

### 1.4 Retry / fallback / model switching

`_Method.__call__` ([llm_client.py:153-195](../src/paperfill/ai/llm_client.py)):

- Primary call; on **any** `Exception` ([line 163](../src/paperfill/ai/llm_client.py)),
  retry **exactly once** on the fallback client. No same-provider retry, no
  backoff, no retry budget.
- The retry **rewrites `model` to `models.get("fallback")`**
  ([lines 182-183](../src/paperfill/ai/llm_client.py)), resolved per call
  ([lines 249-255](../src/paperfill/ai/llm_client.py)). Every other kwarg —
  including image content parts, `response_format`, `temperature`, `extra_body` —
  is forwarded unchanged ([line 179](../src/paperfill/ai/llm_client.py)).
- A fallback failure re-raises ([lines 187-191](../src/paperfill/ai/llm_client.py)).
- If no fallback client exists, the primary error re-raises
  ([lines 164-169](../src/paperfill/ai/llm_client.py)).

**Consequence:** the `fallback` slot serves *all seven* purposes, image payloads
included, despite `needs_vision=False` at
[models.py:70](../src/paperfill/data/models.py). It must be vision-capable and
must accept `response_format: json_schema` (strict) or the detection paths break
on fallback.

Metering: every chat call records purpose, model, provider, token counts,
latency and cost ([llm_client.py:197-230](../src/paperfill/ai/llm_client.py)),
labelled by `call_context` ([llm_client.py:38-68](../src/paperfill/ai/llm_client.py)).

### 1.5 Call sites

| Purpose label | Slot resolved | Defined at | Invoked from |
|---|---|---|---|
| `ocr` | `vision` | [vision_preprocess.py:114-136](../src/paperfill/ai/vision_preprocess.py) | [preprocess.py:1068-1075](../src/paperfill/ai/preprocess.py) |
| `detect` | `detect` | [multimodal_preprocess.py:189-246](../src/paperfill/ai/multimodal_preprocess.py) | [app.py:2256](../src/paperfill/app.py) |
| `detect` (same label, different slot) | `regions` | [candidates.py:391-432](../src/paperfill/ai/candidates.py) | [app.py:2254](../src/paperfill/app.py) |
| `vision_fill` | `vision_pro` / `vision` | [app.py:1131-1172](../src/paperfill/app.py) | [app.py:2407](../src/paperfill/app.py) |
| `text_fill` | `text_fill_pro` / `text_fill` | [app.py:1001-1077](../src/paperfill/app.py) | [app.py:2414](../src/paperfill/app.py) |
| `ask_ai` | `vision_pro` / `vision` | [app.py:1290-1341](../src/paperfill/app.py) | [app.py:2846](../src/paperfill/app.py) |
| `refine` | `text_fill_pro` / `text_fill` | [app.py:1344-1399](../src/paperfill/app.py) | [app.py:2883](../src/paperfill/app.py) |
| `context_image` | `vision` | [context_sources.py:62-92](../src/paperfill/utils/context_sources.py) | [app.py:2352](../src/paperfill/app.py) |

Pro/free selection: `_vision_model` ([app.py:230-231](../src/paperfill/app.py)),
`_ai_model` ([app.py:234-235](../src/paperfill/app.py)). Note `context_image`
hardcodes `models.get("vision")` and never uses `vision_pro`
([context_sources.py:79](../src/paperfill/utils/context_sources.py)).

There are therefore **eight distinct prompt/parser contracts** across
**seven configurable slots**; `vision`/`vision_pro` serve three of them
(`ocr`, `vision_fill`, `ask_ai`) with three different output contracts.

---

## 2. Per-slot specification

---

### 2.1 `ocr` — scanned-page blank detection (slot `vision`)

**Purpose.** For a page that is an image with no usable text layer, locate every
place a student writes and return both the surrounding printed sentence and the
blank's box, so the code can synthesize `Unit`/`Slot` geometry that the text-layer
pipeline could not produce. Triggered by
`page_is_scanned(page) or not page_has_text_layer(page)`
([preprocess.py:1068](../src/paperfill/ai/preprocess.py)) — **only on the
`deterministic` detector path**.

**Input.**
- Image: yes. `page.get_pixmap(dpi=VISION_DPI)`, PNG, no resize, no crop, no
  compression tuning ([vision_preprocess.py:117-118](../src/paperfill/ai/vision_preprocess.py)).
  `VISION_DPI = int(os.environ.get("VISION_DPI", "200"))`
  ([vision_preprocess.py:27](../src/paperfill/ai/vision_preprocess.py)); not set in
  `.env`, so **200 DPI**. US Letter → **1700 × 2200 px** (measured), ~55 KB PNG,
  sent as a `data:image/png;base64` URI.
- No OCR text, no prior model output, no user data. Renders the **rotated** page
  ([vision_preprocess.py:80-91](../src/paperfill/ai/vision_preprocess.py) explains why).
- No `temperature`, no `max_tokens`.

**System prompt** (verbatim, `_SYSTEM`, [vision_preprocess.py:31-59](../src/paperfill/ai/vision_preprocess.py); `_BLANK_TOKEN` is `___`):

```
You analyze a scanned worksheet page image and locate every place a student is expected to write an answer. Return ONLY a JSON object — no prose, no markdown fences.

Shape:
{ "items": [ {
    "kind": "inline" | "open",
    "prompt": "the full sentence or question, with each fill-in blank written as the literal token ___ in reading order",
    "blanks": [ [x0,y0,x1,y1], ... ]
} ] }

Rules:
- Coordinates are normalized floats in [0,1] relative to the image, origin at the TOP-LEFT, as [x0,y0,x1,y1].
- Each blank box bounds the actual blank line/underscore where the answer goes. The number of ___ tokens in 'prompt' MUST equal the length of 'blanks', in the same left-to-right, top-to-bottom order.
- Use kind 'inline' for short fill-in-the-blank lines embedded in a sentence (most worksheet items). Use 'open' only for questions answered in a large empty area with no printed line; give a single box covering that answer area and no ___ token in the prompt.
- Ignore header fields you are unsure about (Nombre, Fecha, Hora) only if they are page chrome; include them as inline items if they are clearly answer lines.
- Do not invent answers. Only locate the blanks and transcribe the surrounding printed text accurately, including Spanish accents.
```

**User prompt** (verbatim, [vision_preprocess.py:129](../src/paperfill/ai/vision_preprocess.py)):

```
Locate every answer blank on this page.
```

followed by the image part.

**Output contract.** `{"items":[{"kind","prompt","blanks"}]}`. Prose-only schema —
`response_format={"type":"json_object"}`
([vision_preprocess.py:134](../src/paperfill/ai/vision_preprocess.py)), **no JSON
schema**.

Coordinate convention, exactly as the parser reads it:
- Order: **`[x0, y0, x1, y1]`** — x first.
- Origin: **top-left**.
- Normalization: **floats in `[0,1]`** relative to the rendered image.
- Not center+w/h. Not y-first. Not 0-1000.
- `_clean_norm_box` ([vision_preprocess.py:68-76](../src/paperfill/ai/vision_preprocess.py))
  coerces to float and sorts each axis, so a reversed corner pair survives but a
  wrong axis *order* does not.
- `_norm_box_to_points` ([vision_preprocess.py:79-91](../src/paperfill/ai/vision_preprocess.py))
  multiplies by `page.rect.width/height` and applies `page.derotation_matrix`.

**Parser.** `json_from_response`
([json_utils.py:7-27](../src/paperfill/utils/json_utils.py)) → `detect_scanned_page`
([vision_preprocess.py:139-224](../src/paperfill/ai/vision_preprocess.py)).
Tolerance:
- `extract_json_object` strips `<think>…</think>`, ` ```json ` fences, and leading
  or trailing prose; falls back to the first balanced `{…}` span
  ([json_utils.py:30-82](../src/paperfill/utils/json_utils.py)). No malformed-JSON
  repair beyond that.
- **Hard-fails** on `finish_reason == "length"`
  ([json_utils.py:21-23](../src/paperfill/utils/json_utils.py)) and on no JSON found
  ([line 26](../src/paperfill/utils/json_utils.py)).
- Non-dict items, unparseable boxes, and items with zero valid boxes are dropped
  silently ([vision_preprocess.py:153-164](../src/paperfill/ai/vision_preprocess.py)).
- The `prompt` blank-token count is **not** validated against `len(blanks)`;
  mismatch degrades to `" "` filler
  ([vision_preprocess.py:195](../src/paperfill/ai/vision_preprocess.py)).

**Axis-swap mitigation.** `_page_is_transposed`
([vision_preprocess.py:101-111](../src/paperfill/ai/vision_preprocess.py)): if ≥5
inline blank boxes and ≥60 % are taller than wide, every box is transposed
`[y0,x0,y1,x1]` ([line 170](../src/paperfill/ai/vision_preprocess.py)). This is a
heuristic patch for models that answer in y-first order — it is not a real
convention adapter and fails below 5 blanks.

**Failure mode.**
- Bad coordinates → answers stamped in the wrong place on the page. **Silent.**
- `items` missing or not a list → prints `[vision] page N: no items in model
  response` and returns `[]` ([lines 147-148](../src/paperfill/ai/vision_preprocess.py)),
  i.e. the page silently yields no blanks.
- Parse failure raises out of `preprocess_pdf` and is caught at
  [app.py:2259-2261](../src/paperfill/app.py) → HTTP 400
  `"could not parse PDF: …"`. **Surfaced, no retry, no model fallback beyond the
  provider fallback in §1.4.**

**Volume.** One call **per scanned page**. Zero calls on a text-layer document.
Sequential — the loop at [preprocess.py:1064](../src/paperfill/ai/preprocess.py) is
not parallelized. Client is built once and reused
([preprocess.py:1070-1071](../src/paperfill/ai/preprocess.py)).

**Token estimate** (estimated, no per-slot counter in code):
- Input: system 1,252 chars ≈ **313 tok** + user 41 chars ≈ **10 tok** + image.
  Image at 1700×2200: ≈ **2,300 tok** (Gemini 768-px crops, 3×3), ≈ **765 tok**
  (OpenAI high-detail tiling), ≈ **2,530 tok** (Anthropic). → **~1.1k–2.6k in**.
- Output: one item per blank, ~120 chars each. A 20-blank page ≈ 2,400 chars
  ≈ **600 tok out**.

---

### 2.2 `detect` — AI Vision detector, anchor-text addressing

**Purpose.** Whole-document call that lists every answer space by **quoting the
printed text next to it**, plus which side the blank is on. No coordinates are
requested; a text resolver ([multimodal_preprocess.py:398-552](../src/paperfill/ai/multimodal_preprocess.py))
finds that string in the PDF char map and derives geometry from it. Selected by
`detector=multimodal|mm|vision2` or `PAPERFILL_DETECTOR`
([app.py:2241-2247, 2256](../src/paperfill/app.py)).

**Input.**
- Image: yes, **all pages in one message**. `MULTIMODAL_INPUT` defaults to
  `"image"` ([multimodal_preprocess.py:152](../src/paperfill/ai/multimodal_preprocess.py));
  each page rendered `get_pixmap(dpi=MULTIMODAL_DPI)` → PNG → base64 data URI,
  interleaved with a `--- PAGE i IMAGE ---` text marker
  ([multimodal_preprocess.py:173-186](../src/paperfill/ai/multimodal_preprocess.py)).
  `MULTIMODAL_DPI` defaults to **150**
  ([line 153](../src/paperfill/ai/multimodal_preprocess.py)); not set in `.env`.
  US Letter → **1275 × 1650 px/page** (measured), ~42 KB PNG. No resize, no crop.
- Alternative `MULTIMODAL_INPUT="pdf"` path: whole PDF via Files API, falling back
  to inline base64 ([lines 156-170](../src/paperfill/ai/multimodal_preprocess.py)),
  plus `extra_body={"plugins":[{"id":"file-parser","pdf":{"engine":"pdf-text"}}]}`
  ([lines 215-216](../src/paperfill/ai/multimodal_preprocess.py)) — an
  OpenRouter-specific key that is sent to the primary provider too.
- OCR text: yes — `page.get_text()` for every page is concatenated into the user
  message ([lines 142-145, 201-209](../src/paperfill/ai/multimodal_preprocess.py)).
  **Uncapped.** Measured: 16,106 chars for an 8-page packet.
- Prior model output: no. User data: no.
- `temperature=0` ([line 224](../src/paperfill/ai/multimodal_preprocess.py)).

**System prompt** (verbatim, `_SYSTEM`, [multimodal_preprocess.py:86-127](../src/paperfill/ai/multimodal_preprocess.py)):

```
You read a worksheet and list every place a student is expected to WRITE an answer. Do not solve anything; only locate blanks.

For each answer space return:
  - page: the 0-based page index it appears on.
  - kind: 'inline' for a short fill-in sitting inside a printed line (a blank line, an underscore run, or empty space after a prompt word/dash); 'open' for a question answered in the large empty area beneath it.
  - anchor_text: the printed text DIRECTLY TOUCHING this blank, transcribed VERBATIM (exact words, casing, punctuation, accents). It is used to find the blank again by exact text search, so it must be the real words next to THIS blank and unique enough to land on it.
  - blank_position: for inline, whether the blank is 'after' or 'before' the anchor_text; for open use 'none'.

Choosing anchor_text (this is the part models get wrong):
- Use the words IMMEDIATELY beside the blank — the word/phrase the blank physically abuts. If the blank is to the LEFT of a word (numbered lists, matching columns, '____ Epididymis'), the anchor is THAT word and blank_position='before'. If the blank follows text ('The capital is ___'), the anchor is the text before it and blank_position='after'.
- NEVER use a shared column header, title, row label, or generic word (e.g. 'Structure', 'Order', 'Answer', 'Name', 'the') as the anchor for a blank that actually sits next to specific content. Each row/item has its OWN distinct text — use that.
- Every item's anchor_text MUST be DIFFERENT. If two blanks would get the same anchor, lengthen each to include neighbouring words until unique. A good anchor is typically 2-6 words.
- Copy the text exactly as printed (a paraphrase will fail to match and the blank is dropped). For 'open', anchor_text is the full question text.

What counts:
- A term/prompt followed by a dash or colon then empty space ('photosynthesis -', 'Capital:') IS an inline blank (write in the space, no printed line needed); anchor='photosynthesis -', position='after'.
- A hyphenated/compound word in running text ('single-eyed', 'well-being') is NOT a blank.
- Headings, titles and instructions are not answer spaces unless they are themselves a labelled fill-in.
- Each distinct blank is its own item, even when several share a line.
```

**User prompt** (verbatim template, [multimodal_preprocess.py:204-209](../src/paperfill/ai/multimodal_preprocess.py)):

```
Locate every answer blank in this worksheet. Pages are 0-indexed; use the page numbers shown below. Use the extracted page text only to transcribe anchor_text accurately; the images are authoritative for layout.

--- PAGE 0 TEXT ---
<page 0 get_text()>

--- PAGE 1 TEXT ---
<page 1 get_text()>
…
```

then, per page: `--- PAGE i IMAGE ---` + image part.

**Output contract.** Provider-enforced strict JSON schema
([multimodal_preprocess.py:232-239](../src/paperfill/ai/multimodal_preprocess.py)),
name `answer_spaces`, `strict: true`. Schema verbatim
([lines 37-83](../src/paperfill/ai/multimodal_preprocess.py)):

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "page":            {"type": "integer"},
          "kind":            {"type": "string", "enum": ["inline", "open"]},
          "anchor_text":     {"type": "string"},
          "blank_position":  {"type": "string", "enum": ["after", "before", "none"]}
        },
        "required": ["page", "kind", "anchor_text", "blank_position"]
      }
    }
  },
  "required": ["items"]
}
```

**No coordinates at all.** The address is a **verbatim string**. Matching rule,
in order ([`_PageIndex.resolve`, multimodal_preprocess.py:527-552](../src/paperfill/ai/multimodal_preprocess.py)):

1. NFC-normalize, fold dash/quote variants, drop zero-width, collapse whitespace,
   lowercase ([lines 257-278](../src/paperfill/ai/multimodal_preprocess.py)); exact
   substring search against the page's flattened char map, rejecting matches cut
   out of the middle of a word (`_token_bounded`,
   [lines 372-381](../src/paperfill/ai/multimodal_preprocess.py)). Anchors shorter
   than 2 chars are refused ([line 386](../src/paperfill/ai/multimodal_preprocess.py)).
2. Alphanumeric-only fuzzy fallback (accents folded, punctuation dropped),
   minimum 3 chars ([lines 543-545](../src/paperfill/ai/multimodal_preprocess.py)).
3. Scattered-token cluster match (`_cluster`,
   [lines 453-525](../src/paperfill/ai/multimodal_preprocess.py)): ≥2 tokens found,
   ≥50 % character coverage, cluster within 55 % page width / 45 % page height of
   the seed, and the anchor's leading token must be present.

Repeated anchors are disambiguated by a reading-order cursor and a used-span set
([lines 423-451](../src/paperfill/ai/multimodal_preprocess.py)).

`page` is read as 0-based; if every value lands in `1..len(doc)` and the 0-based
pass dropped anchors, the whole batch is retried with `shift=1` and the
better-resolving pass wins ([lines 732-813](../src/paperfill/ai/multimodal_preprocess.py)).

**Parser.** `json_from_response` ([line 244](../src/paperfill/ai/multimodal_preprocess.py))
— same strictness as §2.1, including hard-fail on truncation. `items` not a list →
`[]` ([line 246](../src/paperfill/ai/multimodal_preprocess.py)).

**Failure mode.**
- Paraphrased or hallucinated anchor → resolver misses → item dropped with reason
  `anchor_not_found`, logged and returned in `structure["dropped"]`
  ([lines 770-773, 815-817, 951-953](../src/paperfill/ai/multimodal_preprocess.py)).
  **Caught and visible**, but the blank is simply never filled.
- Anchor that resolves to the *wrong* occurrence → answer written in the wrong
  place. **Silent.**
- Parse/schema failure raises → HTTP 400 at [app.py:2259-2261](../src/paperfill/app.py).
  No retry, no alternate-model retry.

**Volume.** **One call per document**, regardless of page count
([multimodal_preprocess.py:723](../src/paperfill/ai/multimodal_preprocess.py)). No
fan-out. Multiple-choice units are added deterministically with no model call
([lines 917-947](../src/paperfill/ai/multimodal_preprocess.py)).

**Token estimate.**
- Input: system 2,263 chars ≈ **566 tok**; page text ≈ **500 tok/page**
  (measured 2,013 chars/page on an 8-page packet); images at 1275×1650 ≈
  **1,550 tok/page** (Gemini), **765 tok/page** (OpenAI), **2,530 tok/page**
  (Anthropic). Single page ≈ **2.6k in**; 8 pages ≈ **17k in** (Gemini).
- Output: ~90 chars/item. 20 items ≈ **450 tok out**; a 100-item packet ≈
  **2,300 tok out**. This is the slot most exposed to the truncation hard-fail.

---

### 2.3 `regions` — code-proposes / model-selects

**Purpose.** Geometry enumerates every place a student *could* write, draws
numbered boxes onto a copy of the PDF, and the model returns the ids of the boxes
that are real answer spaces plus a short prompt naming each item. The model never
supplies a location, only picks from a list. Selected by `detector=regions|region`
([app.py:2244-2245, 2254](../src/paperfill/app.py)).

**Input.**
- Image: yes, **all annotated pages in one message**. `_annotated_doc`
  ([candidates.py:285-303](../src/paperfill/ai/candidates.py)) draws each candidate
  rect plus a filled label box with the `rN` id at 8 pt; rendered
  `get_pixmap(dpi=SELECTION_DPI)` → PNG
  ([candidates.py:477](../src/paperfill/ai/candidates.py)).
  `SELECTION_DPI = int(os.environ.get("REGION_DPI", "150"))`
  ([candidates.py:319](../src/paperfill/ai/candidates.py)); not set in `.env` →
  **150 DPI**, US Letter → **1275 × 1650 px/page**.
- Text: a plain listing of every region id, its page, kind and nearest printed
  label ([candidates.py:400-407](../src/paperfill/ai/candidates.py)). Measured
  sizes: 6 regions / 391 chars on a 1-page vocab sheet; 16 regions / 716 chars on
  an 8-page packet.
- No page `get_text()` dump, no prior model output, no user data.
- `temperature=0` ([candidates.py:417](../src/paperfill/ai/candidates.py)).

**System prompt** (verbatim, `_SELECT_SYSTEM`, [candidates.py:362-388](../src/paperfill/ai/candidates.py)):

```
Each worksheet page image has numbered boxes drawn on it. Every box is a place a student COULD write. Your job is to say which ones a student actually SHOULD write in, and what question each one answers.

The boxes were placed by geometry, not by understanding, so many of them are wrong. Reject any box that is a margin, a gap between sections, whitespace under a heading, empty space on an answer-key or instructions page, or a stray sliver. Keep a box only if a student writes an answer there.

For each box you keep, return its id exactly as printed and a short prompt naming the item it belongs to (the question number and what is being asked, or the label beside the blank). The prompt is what a later step uses to work out the answer, so it must identify the item, but it does not have to restate the whole question.

A box drawn round a coordinate grid is where a graph gets drawn. Keep it if the question asks the student to graph something, and read the axis tick labels to fill in axis_range as [xmin, xmax, ymin, ymax]. Every other box leaves axis_range empty.

Do NOT solve anything. Do not invent ids that are not drawn on the page. If two boxes cover the same answer space, keep the one that fits it better and drop the other. A question answered in the space below it gets the box under it, not the box beside it.
```

**User prompt** (verbatim template, [candidates.py:405-411](../src/paperfill/ai/candidates.py)):

```
Pick the boxes that are real answer spaces.

Boxes drawn on the pages:
r1: page 0, blank, near 'Nombre'
r2: page 0, area, near '1. La capital de España es'
…
```

then, per page: `--- PAGE i ---` + image part.

**Output contract.** Provider-enforced strict JSON schema
([candidates.py:421-426](../src/paperfill/ai/candidates.py)), name `selections`,
`strict: true`. Schema verbatim ([lines 322-359](../src/paperfill/ai/candidates.py)):

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "selections": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "region_id":  {"type": "string"},
          "prompt":     {"type": "string"},
          "axis_range": {"type": "array", "items": {"type": "number"}}
        },
        "required": ["region_id", "prompt", "axis_range"]
      }
    }
  },
  "required": ["selections"]
}
```

Coordinate convention: **none for location** — `region_id` is a verbatim string
matched exactly against `by_id`
([candidates.py:484, 493-495](../src/paperfill/ai/candidates.py)), whitespace
stripped, no normalization. The one numeric array is `axis_range` =
**`[xmin, xmax, ymin, ymax]`** in the graph's own data units, read off tick labels
— *not* a bbox, not normalized, not pixel-space
([candidates.py:344-352](../src/paperfill/ai/candidates.py)).

**Parser.** `json_from_response` ([candidates.py:430](../src/paperfill/ai/candidates.py))
→ `region_preprocess_pdf` ([candidates.py:450-575](../src/paperfill/ai/candidates.py)).
- Unknown `region_id` → dropped with reason `unknown_region`, logged
  ([lines 496-498](../src/paperfill/ai/candidates.py)).
- Duplicate ids ignored after the first ([lines 499-501](../src/paperfill/ai/candidates.py)).
- Empty `prompt` falls back to the geometric label
  ([line 503](../src/paperfill/ai/candidates.py)).
- `axis_range` validated by `_axis_range`
  ([lines 435-447](../src/paperfill/ai/candidates.py)): must be a 4-element numeric
  array with `x_max > x_min` and `y_max > y_min`, else the graph unit is dropped
  (`graph_without_axis_range`, [lines 508-512](../src/paperfill/ai/candidates.py)).

**Failure mode.**
- Over-selecting margins → answers stamped in whitespace. **Silent.**
- Under-selecting → blanks never filled. **Silent** (a region not picked is not a
  drop; it just vanishes).
- Invented ids and bad axis ranges → dropped, logged, counted in
  `dropped_count`. **Caught.**
- Parse failure → raises → HTTP 400 at [app.py:2259-2261](../src/paperfill/app.py).

**Volume.** **One call per document.** MC units added deterministically without a
model call ([candidates.py:545-563](../src/paperfill/ai/candidates.py)).

**Token estimate.**
- Input: system 1,329 chars ≈ **332 tok**; region listing ≈ **100 tok**
  (measured 391 chars, 1 page) to **180 tok** (716 chars, 8 pages); images ≈
  **1,550 tok/page** (Gemini). 1 page ≈ **2.0k in**; 8 pages ≈ **12.9k in**.
- Output: ~60 chars per selection. 15 selections ≈ **230 tok out**.

---

### 2.4 `vision_fill` — answer the page (slots `vision_pro` / `vision`)

**Purpose.** Given one page image and the units detected on that page, produce the
answer text for every slot/unit id. This is the default fill path
(`VISION_FILL = os.environ.get("PAPERFILL_VISION_FILL","1") != "0"`,
[app.py:407](../src/paperfill/app.py)). Detection owns *where*; this owns only
*what*.

**Input.**
- Image: yes, **one page per call**. `_render_page_pngs` →
  `page.get_pixmap(dpi=VISION_FILL_DPI)` → PNG
  ([app.py:1080-1089](../src/paperfill/app.py)).
  `VISION_FILL_DPI = int(os.environ.get("VISION_FILL_DPI","150"))`
  ([app.py:409](../src/paperfill/app.py)); not set in `.env` → **150 DPI**,
  US Letter → **1275 × 1650 px**. No resize, no crop.
- Prior model output: yes, indirectly — the units come from whichever detector ran.
- User data: yes — `instructions` (≤8,000 chars) plus `context` (≤30,000 chars),
  concatenated at [app.py:2392-2398](../src/paperfill/app.py). `context` may itself
  contain the `context_image` slot's output and YouTube transcripts.
- Second pass adds a plain-text list of unanswered ids
  ([app.py:1150-1154](../src/paperfill/app.py)).
- No `temperature`, no `max_tokens`.

**System prompt** (verbatim, `_VISION_FILL_SYSTEM`, [app.py:1092-1128](../src/paperfill/app.py)):

```
You are filling in a worksheet. You are shown ONE worksheet page image AND a JSON list of the 'units' detected on that page. Each unit has an id and a prompt: 'inline_blanks'/'table' prompts contain {{slot_id}} placeholders (answer each slot_id); 'open_response' units are keyed by their answer_key.
'multiple_choice' units carry an 'options' list (each with a 'label' — a letter A/B/C… or a Roman numeral I/II/III… — and text): pick the ONE correct option and return just its label EXACTLY as shown (e.g. "C" or "III") keyed by the unit's answer_key.
Answer EVERY unit in the list — do not skip any. Read the PAGE IMAGE to understand each item: use any answer bank, word box, matching option list, table, diagram or worked example you can see. The image is authoritative; the unit list just tells you which id each answer belongs to.
MATCHING items: when a term has a blank and there is a SEPARATE list of lettered or numbered options (e.g. 'A. to wake up', '1. nucleus'), the answer is the matching option's LABEL — the letter or number — NOT the option's text.
'graph' units are a coordinate grid to plot on: answer with ONE STRING holding the points on the curve as "(x, y)" pairs, e.g. "(-1, 7), (0, 1), (1, -1), (2, 1), (3, 7)" — a string, never a JSON array or a nested list. Give 7 to 15 points spread across the visible x-range, including any intercepts and the vertex, and nothing else — the points are plotted literally where you put them.
Give the ACTUAL answer in the language and format the item calls for (a word, phrase, conjugated verb, letter, number, date, …). Be accurate. Never reply with meta or filler text. For a multi-part answer (point k of n), write a DIFFERENT specific point in each, never repeating.
If the user provides instructions or an answer key, treat those as authoritative and prefer them over your own knowledge.
Write math the way it would be handwritten on the page: √, ∛, π, °, x², and a slash for fractions (5√6/√22). NEVER use LaTeX — no \frac, no \sqrt, no backslash commands, no $…$ and no \(…\) delimiters: the answer is drawn onto the paper exactly as you write it.
Return ONLY a JSON object: {"<slot_or_unit_id>": "<answer>", ...}. No prose, no markdown, no <think> tags. /no_think
```

**User message parts** (in order, [app.py:1143-1156](../src/paperfill/app.py)):

1. Only when instructions are non-empty:
   `"User-provided answer key / instructions (authoritative — prefer over your own knowledge):\n" + instructions`
2. `"Units detected on this page:\n" + <structure_json>` where `structure_json` is
   `strip_bboxes_for_llm` output ([app.py:959-998](../src/paperfill/app.py)) —
   `{"units":[{"unit_id","type","prompt", "slots"|"answer_key"|"options"}]}` with
   all bboxes removed.
3. Retry pass only:
   `"A previous pass left these ids unanswered. They are all answerable from the page — work each one out and return an answer for every id: " + ", ".join(missing_ids)`
4. The page image.

**Output contract.** Flat object `{"<slot_or_unit_id>": "<answer>"}`. Only
`response_format={"type":"json_object"}`
([app.py:1165](../src/paperfill/app.py)) — **no schema**.

Per-type matching rules:
- `inline_blanks` / `table`: keys are `sN` slot ids.
- `open_response` / `graph`: key is the `uN` unit id.
- `multiple_choice`: the value must be the option **label verbatim**. Matched by
  `_match_option` ([render.py:540-556](../src/paperfill/ai/render.py)): uppercase,
  take the leading `[A-Z]+` run, exact map lookup, then first-character lookup.
  Tolerates `"C."`, `"c)"`, `"C) 2(3m-5n)"`, `"III only"`.
- `graph`: **one string** of `"(x, y)"` pairs. `parse_points`
  ([render.py:497-517](../src/paperfill/ai/render.py)) regex-matches pairs; if none,
  accepts an even run of ≥4 loose numbers and pairs them positionally.
  `plot_point` ([render.py:520-537](../src/paperfill/ai/render.py)) maps them via
  the detected axis origin and silently drops any point outside the plot.

**Parser.** `extract_json_object` → `_flatten_answers`
([app.py:1167-1168](../src/paperfill/app.py)). **Markedly more permissive than the
detection paths — it never raises.**
- No truncation check: `json_from_response` is *not* used here. A response cut off
  at the token cap parses to `{}` or a partial object.
- `_flatten_answers` ([app.py:1253-1270](../src/paperfill/app.py)) accepts flat,
  nested (`{"u1":{"s1":…}}`), and composite keys; `_normalize_key`
  ([app.py:1240-1250](../src/paperfill/app.py)) extracts a trailing `sN`/`uN` from
  `"u1-s2"`, `"unit1.s4"`, `"slot_s5"`.
- `_answer_text` ([app.py:1273-1287](../src/paperfill/app.py)) coerces bool → `"True"/"False"`,
  int/float → `str`, list → comma-joined, and pipes every string through
  `plain_math` ([utils/plain_math.py:76](../src/paperfill/utils/plain_math.py)),
  which rewrites LaTeX into Unicode glyphs.
- Empty result → prints a warning with the first 400 chars
  ([app.py:1169-1171](../src/paperfill/app.py)) and returns `{}`.

**Failure mode.**
- Wrong answer text → written onto the page as-is. **Silent.**
- Missing ids → one automatic **retry call for that page** naming the missing ids
  ([app.py:1210-1219](../src/paperfill/app.py)). Ids still missing after the retry
  are left blank, silently.
- Exception or empty result for the **whole document** → caught at
  [app.py:2409-2410](../src/paperfill/app.py), logged, and the fill falls back to
  the **`text_fill` slot** ([app.py:2412-2415](../src/paperfill/app.py)). Which
  path ran is recorded as `job["fill_path"]` ([app.py:2427](../src/paperfill/app.py)).
  If `text_fill` also raises → HTTP 502 ([app.py:2416-2417](../src/paperfill/app.py)).
- Note the fallback triggers on `not answers` for the whole document, so a
  partially-answered document does **not** fall back.

**Volume.** **1–2 calls per page that has units.** Pages fan out across
`ThreadPoolExecutor(max_workers=min(4, len(pages)))`
([app.py:1224](../src/paperfill/app.py)). Worst case for a 15-page free-tier
document: **30 calls**. Pages with no detected units are skipped
([app.py:1199-1201, 1222](../src/paperfill/app.py)).

**Token estimate.**
- **Measured**: the `usage.py` docstring states this call "averages about 1,560
  tokens per page it fills" ([usage.py:20-22](../src/paperfill/data/usage.py)) —
  prompt + output combined, from provider usage figures. This is the only
  measured number in the codebase.
- Estimated breakdown per call: system 2,230 chars ≈ **558 tok**; unit JSON
  ≈ **130–670 tok/page** (measured 509 chars for 4 units, 5,359 chars for 36
  units over 2 pages); instructions+context up to 38,000 chars ≈ **9,500 tok** if
  the user attaches material; image ≈ **1,550 tok** (Gemini) / **765 tok**
  (OpenAI) / **2,530 tok** (Anthropic).
- Output: one short string per id. 10 units ≈ 400 chars ≈ **100 tok**; a dense
  36-unit page ≈ **500 tok**. Open-response and graph units push this higher.

---

### 2.5 `text_fill` — answer from transcribed text only (slots `text_fill_pro` / `text_fill`)

**Purpose.** Same job as `vision_fill` with **no image**: answer every unit from
the stripped structure alone. Runs when `VISION_FILL` is off, or as the automatic
fallback when the vision fill errors or returns nothing
([app.py:2404-2417](../src/paperfill/app.py)).

**Input.**
- Image: **no**.
- User data: same `instructions` + `context` string as §2.4.
- Whole document in **one** call — not split by page.
- No `temperature`, no `max_tokens`.

**System prompt** (verbatim, [app.py:1012-1048](../src/paperfill/app.py)):

```
You are filling in a worksheet PDF. You receive a list of units. For each unit:
  - 'inline_blanks' or 'table': the prompt contains {{slot_id}}     placeholders. Return the answer for each slot_id.
  - 'open_response': the prompt is a question. Return one answer     keyed by the unit's answer_key, kept to a few sentences.
  - 'multiple_choice': the prompt is a question with an 'options' list,     each option having a 'label' (a letter A, B, C… or a Roman numeral     I, II, III…) and text. Pick the ONE correct option and return just     its label EXACTLY as shown (e.g. "C" or "III") keyed by the     unit's answer_key.
Use the context in each prompt to figure out what kind of answer fits (a single word, a phrase, a conjugated verb form, a name, a date, etc.). Be accurate. If you genuinely don't know something factual (e.g. the user's name), pick a reasonable placeholder like 'Student'.
Always give the ACTUAL answer/definition. Never reply with meta or filler text such as 'Answer the prompt based on your situation' or 'Complete the prompt with relevant information' — for a definition question, write the real definition.
When a prompt is marked as a multi-part answer (point k of n), the units that share that question together form ONE list answer: write a DIFFERENT, specific point in each (e.g. the five components of SMART goals, or distinct functions of the Federal Reserve) and never repeat the same sentence across them.
If the user provides instructions or an answer key, treat those as authoritative and prefer them over your own knowledge.
Write math the way it would be handwritten on the page: √, ∛, π, °, x², and a slash for fractions (5√6/√22). NEVER use LaTeX — no \frac, no \sqrt, no backslash commands, no $…$ and no \(…\) delimiters: the answer is drawn onto the paper exactly as you write it.
Return ONLY a JSON object: {"<slot_or_unit_id>": "<answer>", ...}. No prose, no markdown, no <think> tags, no explanations — JSON only. /no_think
```

(The doubled spaces after `-` are literal — the source concatenates lines that
each begin with `"    "` at [app.py:1016, 1018, 1021-1023](../src/paperfill/app.py).)

**User prompt** (verbatim, [app.py:1053-1061](../src/paperfill/app.py)) — with
instructions:

```
User-provided answer key / instructions (use these as the authoritative source — prefer them over your own knowledge):
<instructions>

Worksheet to fill:
<structure_json>
```

Without instructions the user message is **just** `<structure_json>`.

**Output contract.** Identical to §2.4 (flat `{id: answer}`, `json_object` mode at
[app.py:1070](../src/paperfill/app.py), no schema), **except** there is no image,
so `graph` units are answered purely from the axis range that
`region_preprocess_pdf` baked into `prompt_text`
([candidates.py:519-521](../src/paperfill/ai/candidates.py), noted at
[app.py:993-996](../src/paperfill/app.py)).

**Parser.** `extract_json_object` → `_flatten_answers`
([app.py:1072-1077](../src/paperfill/app.py)). Same permissiveness as §2.4; never
raises; warns with the first 800 chars on an empty result
([app.py:1076](../src/paperfill/app.py)).

**Failure mode.** Wrong answers are silent. An empty result here is terminal —
this *is* the fallback, so `answers` stays `{}` and the page is rendered with no
overlays. A raised exception → HTTP 502
([app.py:2416-2417](../src/paperfill/app.py)).

**Volume.** **One call per document**, and only on the fallback branch. Zero calls
when the vision fill succeeds.

**Token estimate.**
- Input: system 1,971 chars ≈ **493 tok**; structure JSON for the whole document
  ≈ **670 tok/page** (measured 5,359 chars for a 2-page, 36-unit sheet);
  instructions+context up to ≈ **9,500 tok**.
- Output: same per-unit size as §2.4 but for **every** page in one response —
  a 15-page sheet is where the missing truncation check bites hardest.

---

### 2.6 `ask_ai` — answer one hand-snipped item (slots `vision_pro` / `vision`)

**Purpose.** The user drags a box around a question the fill left blank; the model
reads that crop and returns only the answer to write in
([app.py:1290-1341](../src/paperfill/app.py), route at
[app.py:2800-2851](../src/paperfill/app.py)).

**Input.**
- Image: yes, a **crop**. `page.get_pixmap(dpi=VISION_DPI, clip=…)` with a 4-point
  pad on each side ([app.py:2834-2839](../src/paperfill/app.py)). Note this uses
  `VISION_DPI` (**200**, imported from
  [vision_preprocess.py:27](../src/paperfill/ai/vision_preprocess.py) at
  [app.py:66](../src/paperfill/app.py)), not `VISION_FILL_DPI`. A 300 × 60 pt
  selection → **833 × 167 px**. Minimum selection 2 × 2 pt
  ([app.py:2825-2826](../src/paperfill/app.py)).
- User data: `job["fill_instructions"]`, truncated to `SNIP_REF_MAX = 12000` chars
  ([app.py:400, 1328, 2432](../src/paperfill/app.py)).
- No `temperature`, no `max_tokens`.

**System prompt** (verbatim, [app.py:1301-1317](../src/paperfill/app.py)):

```
You are helping a student fill in a worksheet. You are shown a cropped screenshot of ONE worksheet item (a fill-in-the-blank, a short question, or a prompt) that was left unanswered. Read it and return ONLY the answer that should be written in — do not restate the question, add a label, or explain. For a fill-in-the-blank give just the word or phrase; for a short-answer question give a concise answer (a few sentences at most). If the user supplied an answer key or notes, prefer them over your own knowledge.
Write math the way it would be handwritten on the page: √, ∛, π, °, x², and a slash for fractions (5√6/√22). NEVER use LaTeX — no \frac, no \sqrt, no backslash commands, no $…$ and no \(…\) delimiters: the answer is drawn onto the paper exactly as you write it.
Return ONLY a JSON object: {"answer": "<text>"}. No prose, no markdown, no <think> tags. /no_think
```

**User message parts** ([app.py:1319-1329](../src/paperfill/app.py)), in final order:

1. Only when reference text exists (inserted at index 0):
   `"Answer key / reference material the user provided (prefer it over your own knowledge):\n" + instructions[:12000]`
2. `"Answer this worksheet item."`
3. The cropped image.

**Output contract.** `{"answer": "<text>"}`. `response_format={"type":"json_object"}`
([app.py:1338](../src/paperfill/app.py)), no schema. No coordinates.

**Parser.** `extract_json_object(content).get("answer","")` → `_answer_text`
([app.py:1340-1341](../src/paperfill/app.py)). Same coercions and `plain_math`
pass as §2.4. A missing `answer` key yields `""`.

**Failure mode.** Empty or wrong answer → the returned string is handed straight
back to the frontend ([app.py:2851](../src/paperfill/app.py)); an empty string is
**not** treated as an error here (unlike `refine`). An exception → HTTP 502
`"vision call failed: …"` ([app.py:2848-2849](../src/paperfill/app.py)).

**Volume.** **One call per user snip action.** Zero per page or per document
during a normal fill. Unbounded in principle; gated only by the free-tier credit
check ([app.py:2800-2802](../src/paperfill/app.py)).

**Token estimate.**
- Input: system 873 chars ≈ **218 tok**; reference text up to 12,000 chars ≈
  **3,000 tok**; crop image ≈ **500 tok** (Gemini, 2 crops) / **255 tok**
  (OpenAI, 1 tile) for a typical single-item selection.
- Output: **10–120 tok**.

---

### 2.7 `refine` — rewrite one box (slots `text_fill_pro` / `text_fill`)

**Purpose.** The editor's floating toolbar asks for a shorter, longer, or
free-text-instructed rewrite of one already-written answer
([app.py:1344-1399](../src/paperfill/app.py), route at
[app.py:2854-2891](../src/paperfill/app.py)).

**Input.**
- Image: **no**.
- Prior model output: yes — the current box text.
- User data: `mode` ∈ `shorten|lengthen|else`, a free-text `instruction`
  (≤2,000 chars, [app.py:2876](../src/paperfill/app.py)), and
  `job["fill_instructions"]` truncated to 12,000 chars
  ([app.py:1381](../src/paperfill/app.py)).
- No `temperature`, no `max_tokens`.

**System prompt** (verbatim, [app.py:1365-1375](../src/paperfill/app.py)):

```
You are editing a single answer a student wrote in one box of a worksheet. You are given the current text and an instruction for how to change it. Apply the instruction and return ONLY the rewritten text that should replace what's in the box — do not restate the question, add a label or quotes, or explain. Keep it factually correct. If the user supplied an answer key or notes, prefer them over your own knowledge. Return ONLY a JSON object: {"text": "<rewritten text>"}. No prose, no markdown, no <think> tags. /no_think
```

**User prompt** (verbatim assembly, [app.py:1377-1384](../src/paperfill/app.py)) —
blocks joined by a blank line:

```
Answer key / reference material the answer is based on (stay consistent with it):
<ref[:12000]>

Instruction: <directive>

Current text:
<text>
```

`<directive>` is one of ([app.py:1356-1363](../src/paperfill/app.py)):
- `shorten` → `Make this text much more shorter and more concise while keeping the same meaning and the same answer.`
- `lengthen` → `Make this text longer and more detailed while keeping it accurate and on-topic.`
- `else` → the user's own instruction, or `Rewrite this text.` if blank.

**Output contract.** `{"text": "<rewritten text>"}`. `json_object` mode
([app.py:1393](../src/paperfill/app.py)), no schema.

**Parser.** `extract_json_object(content).get("text","")`
([app.py:1396-1399](../src/paperfill/app.py)). Numbers coerced to `str`; anything
else → `""`. **Note:** unlike every other answer path, this one does **not** run
`plain_math`, so LaTeX that survives the prompt reaches the page verbatim.

**Failure mode.** Empty result → HTTP 502 `"couldn't rewrite that text"`
([app.py:2888-2889](../src/paperfill/app.py)). Exception → HTTP 502
`"LLM call failed: …"` ([app.py:2886-2887](../src/paperfill/app.py)). Both
surfaced.

**Volume.** **One call per toolbar edit.** Zero during a normal fill.

**Token estimate.**
- Input: system 523 chars ≈ **131 tok**; reference up to 12,000 chars ≈
  **3,000 tok**; current text typically **20–150 tok**.
- Output: **20–300 tok**.

---

### 2.8 `context_image` — OCR an attached reference image (slot `vision`, always free-tier)

**Purpose.** A user attaches a photo or screenshot as reference material; this
transcribes or describes it so the text can be folded into `instructions`
([context_sources.py:62-92](../src/paperfill/utils/context_sources.py), route at
[app.py:2330-2369](../src/paperfill/app.py)).

**Input.**
- Image: yes — **the uploaded file's raw bytes**, base64-encoded as-is
  ([context_sources.py:76](../src/paperfill/utils/context_sources.py)). No render,
  no DPI, no resize, no re-compression, no crop. Pixel count is whatever the user
  uploaded — **unknown and unbounded from the code**; MIME is derived from the
  extension ([context_sources.py:112](../src/paperfill/utils/context_sources.py)),
  accepted extensions at
  [context_sources.py:31](../src/paperfill/utils/context_sources.py).
- **No system message.** Single user message.
- No `response_format`, no `temperature`, no `max_tokens`.

**User prompt** (verbatim, [context_sources.py:83-87](../src/paperfill/utils/context_sources.py)):

```
Transcribe all text in this image verbatim. If it is a diagram or photo with little text, describe what it shows in a few sentences. Output only the transcription/description.
```

**Output contract.** **Free text.** No JSON, no schema, no coordinates.

**Parser.** `resp.choices[0].message.content or ""`
([context_sources.py:92](../src/paperfill/utils/context_sources.py)), then
truncated to `MAX_FILE_CHARS = 6000`
([context_sources.py:27, 118](../src/paperfill/utils/context_sources.py)). Nothing
is validated. `<think>` blocks and markdown fences are **not** stripped on this
path — they would flow into the fill prompt as reference material.

**Failure mode.** Any exception is swallowed and replaced with a marker string
`"[could not read <filename>: <e>]"`
([context_sources.py:116-117](../src/paperfill/utils/context_sources.py)), which is
then sent to the fill model as if it were reference material. A missing API key
yields `"[image attached, but no API key configured to read it]"`
([context_sources.py:74-75](../src/paperfill/utils/context_sources.py)). **Fully
silent** — the user sees a source with a small `chars` count, nothing more.

**Volume.** **One call per uploaded reference image.** Zero for PDF/text
attachments (extracted locally,
[context_sources.py:44-59](../src/paperfill/utils/context_sources.py)).

**Billing note:** [app.py:2352](../src/paperfill/app.py) calls
`extract_file_text(f.filename, f.read())` without `user_key` or `is_pro`, so these
tokens are metered to the dashboard but **never debited** against the free-tier
credit budget ([llm_client.py:203-213](../src/paperfill/ai/llm_client.py),
[context_sources.py:101-103](../src/paperfill/utils/context_sources.py)).

**Token estimate.**
- Input: prompt 175 chars ≈ **44 tok** + image, **unknown** (user-supplied
  dimensions). A phone photo at 4032 × 3024 would be ≈ **6,700 tok** on Gemini
  tiling.
- Output: capped downstream at 6,000 chars ≈ **1,500 tok**, but nothing caps the
  model — `max_tokens` is not set.

---

## 3. Global questions

### 3.1 Which slots need provider-side structured output?

| Slot / purpose | Enforcement used | Survives a model that cannot enforce JSON? |
|---|---|---|
| `detect` | `json_schema`, `strict: true` ([multimodal_preprocess.py:232-239](../src/paperfill/ai/multimodal_preprocess.py)) | **Partly.** `extract_json_object` would still recover a well-formed object from fenced/prosey output, but `additionalProperties:false` + enums are the only thing keeping `blank_position` and `kind` in-domain. A stray value silently defaults (`position not in ("after","before") → "after"`, [line 884](../src/paperfill/ai/multimodal_preprocess.py)). |
| `regions` | `json_schema`, `strict: true` ([candidates.py:421-426](../src/paperfill/ai/candidates.py)) | **Yes, mostly.** Unknown ids and bad `axis_range` are validated in code ([lines 493-498, 435-447](../src/paperfill/ai/candidates.py)), so a loose model degrades to dropped items rather than corruption. |
| `ocr` | `json_object` only ([vision_preprocess.py:134](../src/paperfill/ai/vision_preprocess.py)) | **Yes** — already free-text-tolerant. |
| `vision_fill`, `text_fill`, `ask_ai`, `refine` | `json_object` only ([app.py:1165, 1070, 1338, 1393](../src/paperfill/app.py)) | **Yes.** `extract_json_object` strips fences, `<think>` blocks and surrounding prose, and `_flatten_answers` normalizes key shapes. |
| `context_image` | none | **Yes** — free text is the contract. |
| `fallback` | inherits whatever the failed call sent | **No.** It receives `json_schema`/`strict` verbatim when it stands in for `detect` or `regions` ([llm_client.py:179](../src/paperfill/ai/llm_client.py)). A fallback model that rejects `json_schema` turns a primary outage into a hard 400. |

Every path except `context_image` requires at least `json_object` support or the
ability to be coaxed into a bare object by prompt alone.

### 3.2 Which slots need native bounding-box grounding?

**Exactly one: `ocr`** ([vision_preprocess.py:36-46](../src/paperfill/ai/vision_preprocess.py)).
It is the only slot that asks the model to emit coordinates, and those coordinates
are used directly as page geometry.

Everything else:
- `detect` — reading + verbatim transcription. Localization is done by string
  search in code ([multimodal_preprocess.py:527-552](../src/paperfill/ai/multimodal_preprocess.py)).
- `regions` — multiple choice over boxes already drawn on the image; the model
  reads a printed id ([candidates.py:285-303](../src/paperfill/ai/candidates.py)).
  Needs OCR-quality small-text reading (8 pt labels,
  [candidates.py:301](../src/paperfill/ai/candidates.py)), not grounding.
- `vision_fill`, `ask_ai`, `context_image` — reading and answering.
- `text_fill`, `refine` — no image at all.

`regions` also needs one numeric reading skill that is not grounding: reading axis
tick labels off a plotted grid ([candidates.py:379-382](../src/paperfill/ai/candidates.py)).

### 3.3 Is the coordinate convention hardcoded to one model family?

**Yes, in the `ocr` slot only.** The prompt demands
`[x0, y0, x1, y1]`, top-left origin, floats in `[0,1]`
([vision_preprocess.py:44-46](../src/paperfill/ai/vision_preprocess.py)). That is
the OpenAI-ish convention. A Gemini-family model that answers in its native
`[y_min, x_min, y_max, x_max]` normalized to `0-1000` produces boxes that are both
axis-swapped and 1000× too large.

Functions that would have to change to switch families:

| Function | File:lines | Why |
|---|---|---|
| `_SYSTEM` | [vision_preprocess.py:31-59](../src/paperfill/ai/vision_preprocess.py) | States the order, origin and range. |
| `_clean_norm_box` | [vision_preprocess.py:68-76](../src/paperfill/ai/vision_preprocess.py) | Assumes `(x0,y0,x1,y1)` unpacking and `[0,1]` scale; would need a `0-1000` divisor and a y-first reorder. |
| `_norm_box_to_points` | [vision_preprocess.py:79-91](../src/paperfill/ai/vision_preprocess.py) | Multiplies index 0/2 by width and 1/3 by height. |
| `_page_is_transposed` | [vision_preprocess.py:101-111](../src/paperfill/ai/vision_preprocess.py) | Exists only to paper over this exact mismatch. With a real adapter it becomes wrong — it would flip correctly-ordered boxes on a page whose blanks genuinely are tall. |

No other slot is affected. `_axis_range` ([candidates.py:435-447](../src/paperfill/ai/candidates.py))
parses `[xmin,xmax,ymin,ymax]` but that is graph data units, not image space, and is
family-independent.

### 3.4 Are images sent to text-only slots, or more image tokens than necessary?

**No image ever reaches a text-only slot on the primary path.** `text_fill`
([app.py:1064-1071](../src/paperfill/app.py)) and `refine`
([app.py:1387-1394](../src/paperfill/app.py)) send text-only messages.

**But the `fallback` slot is declared `needs_vision=False`
([models.py:70](../src/paperfill/data/models.py)) and does receive images** —
`_Method.__call__` forwards the original `messages` unchanged when it retries
([llm_client.py:179-186](../src/paperfill/ai/llm_client.py)). It is currently set
to `google/gemini-3.5-flash`, which is vision-capable, so this works today; the
metadata is what is wrong, not the behaviour.

Wasted image tokens:

1. **`ocr` renders at 200 DPI** ([vision_preprocess.py:27](../src/paperfill/ai/vision_preprocess.py))
   → 1700 × 2200 px. For OpenAI- and Anthropic-style preprocessing this is pure
   waste: both downscale to the same working size a 150 DPI render would produce
   (768 × 994 and 1212 × 1568 respectively), so the extra 78 % of pixels buys
   nothing. For Gemini tiling it is a real 50 % increase (9 crops vs 6).
2. **`ask_ai` also uses 200 DPI** ([app.py:2839](../src/paperfill/app.py)) on a
   crop. Here the higher DPI is defensible — small crops are where resolution
   actually matters.
3. **`detect` and `regions` send every page in one request**
   ([multimodal_preprocess.py:218](../src/paperfill/ai/multimodal_preprocess.py),
   [candidates.py:408-411](../src/paperfill/ai/candidates.py)) with no page cap
   below `MAX_PDF_PAGES = 50` ([app.py:101](../src/paperfill/app.py)).
4. **`detect` sends both the full page image and the full page text**
   ([multimodal_preprocess.py:201-209](../src/paperfill/ai/multimodal_preprocess.py)),
   uncapped. Deliberate — the text is a transcription aid — but it roughly
   doubles the non-image input.
5. **`vision_fill`'s retry re-sends the same page image**
   ([app.py:1215-1217](../src/paperfill/app.py)). Unavoidable given the API, but it
   doubles that page's image cost.
6. **`context_image` sends the user's file at native resolution**
   ([context_sources.py:76](../src/paperfill/utils/context_sources.py)) with no
   downscale. A phone photo costs several thousand tokens to extract at most 6,000
   characters.

### 3.5 Prompt caching, batching, request deduplication?

**None of the three.**

- No `cache_control`, no `prompt_cache_key`, no provider cache hint anywhere. Grep
  for `cache` in `src/paperfill/ai/` returns only the model-catalog TTL cache
  ([models.py:130-155](../src/paperfill/data/models.py)) and the `_PageIndex`
  per-pass caches ([multimodal_preprocess.py:752-753](../src/paperfill/ai/multimodal_preprocess.py)).
- No Batch API use. `client.files.create` is used only to upload a PDF for the
  `MULTIMODAL_INPUT="pdf"` variant
  ([multimodal_preprocess.py:161](../src/paperfill/ai/multimodal_preprocess.py)).
- No deduplication. Re-uploading the same PDF re-runs detection; re-filling re-runs
  every page. `_openai_client` is memoized ([app.py:609-615](../src/paperfill/app.py))
  but that caches the HTTP client, not responses.
- The one thing that *is* concurrent is `vision_fill`'s per-page fan-out
  ([app.py:1224](../src/paperfill/app.py)), which is parallelism, not batching.

`vision_fill` has the strongest caching case that is currently unexploited: its
558-token system prompt plus the user's up-to-9,500-token instructions block is
identical across every page call of a document.

### 3.6 Largest realistic request

The `detect` slot on a 15-page document (the Free-tier ceiling,
`FREE_MAX_PDF_PAGES = 15`, [app.py:106](../src/paperfill/app.py)):

| Component | Tokens |
|---|---|
| System prompt | 566 |
| Page text, 15 × ~500 | 7,500 |
| Page images, 15 × 1,550 (Gemini tiling at 150 DPI) | 23,250 |
| **Input total** | **~31,300** |
| Output, ~100 items × ~90 chars | ~2,300 |

At the hard ceiling `MAX_PDF_PAGES = 50` ([app.py:101](../src/paperfill/app.py))
this becomes **~104,000 input tokens**. On Anthropic-style image accounting the
50-page case is ~127,000.

**A 128k context window covers the realistic worst case; anything below ~64k will
fail on large packets via `detect` or `regions`.** Every other slot is far smaller:
`vision_fill` peaks around **12k input** per call (image + units + a full 38,000-char
context block).

### 3.7 Configured model IDs and base URLs

No `ai_models.json` exists, so every slot resolves via env var or inheritance.
From `.env` (local checkout; **the production values on
`paperfill.hackclub.app` are not in this repo and may differ**):

| Slot | Resolves to | Via |
|---|---|---|
| `vision` | `google/gemini-3-flash-preview` | `VISION_MODEL` (`.env:3`) |
| `vision_pro` | `google/gemini-3.5-flash` | `VISION_MODEL_PRO` (`.env:5`) |
| `detect` | `google/gemini-3-flash-preview` | **inherited** from `vision` — `MULTIMODAL_MODEL` unset |
| `regions` | `google/gemini-3-flash-preview` | **inherited** from `vision` — `REGION_MODEL` unset |
| `text_fill` | `google/gemini-3-flash-preview` | `AI_MODEL` (`.env:4`) |
| `text_fill_pro` | `google/gemini-3.5-flash` | `AI_MODEL_PRO` (`.env:6`) |
| `fallback` | `google/gemini-3.5-flash` | `OPENROUTER_MODEL` (`.env:10`) |

`DEFAULT_MODEL = "openai/gpt-5.5"` ([models.py:33](../src/paperfill/data/models.py))
is unreachable with this `.env`.

Base URL selection:
- **Primary → HCAI proxy.** `AI_BASE_URL` is **not set** in `.env`, so
  [llm_client.py:270-272](../src/paperfill/ai/llm_client.py) falls through to the
  literal default `https://ai.hackclub.com/proxy/v1`. Every call starts here.
- **Fallback → OpenRouter.** `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
  (`.env:9`), read at [llm_client.py:279](../src/paperfill/ai/llm_client.py). Built
  only if `OPENROUTER_API_KEY` is present
  ([llm_client.py:274-276](../src/paperfill/ai/llm_client.py)).
- The **only** thing that routes a call to OpenRouter is a primary exception
  ([llm_client.py:163](../src/paperfill/ai/llm_client.py)). There is no per-slot
  provider choice, no cost-based routing, no health check.

Rate card ([ai_rates.json](../ai_rates.json), defaults at
[costs.py:39-42](../src/paperfill/data/costs.py)): primary is billed at **$0**
(`_provider_primary_is_free: true`, [costs.py:103-104](../src/paperfill/data/costs.py));
`google/gemini-3-flash-preview` $0.50/$3.00 per 1M in/out;
`google/gemini-3.5-flash` $1.50/$9.00. A model with no rate returns `None`
("uncosted"), never `0.0` ([costs.py:105-107](../src/paperfill/data/costs.py)).

The $3/day Hack Club budget is modelled at
[stats.py:338-382](../src/paperfill/data/stats.py) with
`hack_club_cap_usd: float = 3.0` ([stats.py:535](../src/paperfill/data/stats.py)),
which deliberately prices primary-provider calls against the rate card even though
they cost *us* nothing.

---

## 4. Cost model

### Assumptions

1. **Typical document**: 1 page, US Letter (612 × 792 pt), text-layer PDF (not
   scanned), ~10 detected units / ~8 slots. Grounded in the sample worksheets:
   `vocab_dash.pdf` = 4 units, `synthetic_worksheet.pdf` = 6 units,
   `dle_p01.pdf` = 36 units over 2 pages.
2. **Default configuration**: `PAPERFILL_DETECTOR` unset → `deterministic`
   detector, so **no `detect` and no `regions` call**. `PAPERFILL_VISION_FILL`
   unset → **`vision_fill` runs, `text_fill` does not**.
3. **Image tokens** use Gemini-style 768-px tiling at 258 tok/crop, matching the
   configured `google/gemini-*` models. 150 DPI Letter = 1275 × 1650 px = 2 × 3 =
   6 crops = **1,548 tok**. 200 DPI Letter = 1700 × 2200 = 3 × 3 = 9 crops =
   **2,322 tok**. For an OpenAI-family candidate substitute **765 tok** for either;
   for Anthropic, **~2,530 tok**.
4. **Text tokens** at 4 chars/token. Prompt char counts are measured from source.
5. **No attached reference material** (no `context_image`, no instructions block).
6. **`vision_fill` retry fires on 30 % of pages** — the code retries whenever any
   id is unanswered ([app.py:1210-1219](../src/paperfill/app.py)); the real rate is
   **unknown**, so this is a stated guess, not a measurement.
7. `ask_ai` and `refine` are user-initiated and excluded from the per-page baseline.

### Per-page table — default path (deterministic detector + vision fill)

| Slot | Calls/page | Input tok/call | Output tok/call | **Input/page** | **Output/page** |
|---|---|---|---|---|---|
| `vision_fill` (`vision`/`vision_pro`) | 1.0 | 2,300 | 150 | **2,300** | **150** |
| `vision_fill` retry | 0.3 | 2,150 | 60 | **645** | **18** |
| `ocr` (`vision`) | 0 (text-layer page) | — | — | **0** | **0** |
| `detect` | 0 (not selected) | — | — | **0** | **0** |
| `regions` | 0 (not selected) | — | — | **0** | **0** |
| `text_fill` | 0 (vision succeeded) | — | — | **0** | **0** |
| **Total** | | | | **~2,950** | **~170** |

`vision_fill` input = 558 (system) + 200 (unit JSON, 10 units) + 1,548 (image).
Cross-check: 2,950 + 170 = 3,120 tok/page against the measured "~1,560 tokens per
page" at [usage.py:20-22](../src/paperfill/data/usage.py). The measurement is
**~2× lower** than this estimate — consistent with the measured average being taken
over pages with fewer units, a lower effective image-token rate on the proxy, or a
retry rate well under 30 %. Treat 1,560 as the empirical floor and 3,120 as the
pessimistic ceiling.

### Per-page table — alternate paths

Each of these replaces rows above; they do not stack.

| Variant | Slot | Calls | Input/page | Output/page |
|---|---|---|---|---|
| Scanned page (auto) | `ocr` | 1/page | **2,650** (313 sys + 10 user + 2,322 img @200 DPI) | **600** (20 blanks) |
| `detector=multimodal` | `detect` | 1/**document** | **2,600/page** (566 sys ÷ pages + 500 text + 1,548 img) | **450/page** |
| `detector=regions` | `regions` | 1/**document** | **2,000/page** (332 sys ÷ pages + 100 listing + 1,548 img) | **230/page** |
| Vision fill failed | `text_fill` | 1/**document** | **1,200/page** (493 sys ÷ pages + 670 unit JSON) | **170/page** |
| Reference image attached | `context_image` | 1/image | **unknown** (user-supplied resolution) + 44 | ≤**1,500** |
| User snips an item | `ask_ai` | 1/snip | **720** (218 sys + 500 crop img) | **60** |
| User edits a box | `refine` | 1/edit | **200** | **150** |

If the user attaches instructions or context, add up to **9,500 input tokens per
`vision_fill` call** — i.e. up to **12,350 tok/page** on a page that also retries.
This is the single largest swing factor in the model and is entirely user-driven.

### Computing a cost

For a candidate at `$P_in` and `$P_out` per 1M tokens, default path:

```
cost_per_page ≈ (2950 / 1e6) * P_in + (170 / 1e6) * P_out
```

At the currently configured Free-tier `google/gemini-3-flash-preview`
($0.50 / $3.00): **~$0.0020/page**. At the Pro `google/gemini-3.5-flash`
($1.50 / $9.00): **~$0.0059/page**.

Against the $3/day Hack Club budget ([stats.py:535](../src/paperfill/data/stats.py)),
those work out to roughly **1,500 pages/day** at Free-tier pricing and **510
pages/day** at Pro pricing — before any `ask_ai`, `refine`, or `context_image`
traffic. The free-tier credit meter is the practical brake: `FREE_DAILY_CREDITS = 10`
at `CREDIT_TOKENS = 1000` ([usage.py:42-43](../src/paperfill/data/usage.py)) caps a
Free account at **10,000 tokens/day**, i.e. ~3–6 pages, and Pro accounts are
unmetered ([llm_client.py:203-213](../src/paperfill/ai/llm_client.py)).
