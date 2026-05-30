# Paperfill

AI worksheet filler. Upload a PDF with fill-in-the-blanks → AI answers go in
the right slots → download the filled PDF.

## How it works (zero coordinates ever touch the LLM)

1. **Preprocessor** (`preprocess.py`) walks the PDF with PyMuPDF and emits
   a JSON structure: `inline_blanks` (sentence with underscore runs),
   `open_response` (numbered question + empty space), `table` (grid with
   blanks per cell). Every slot has a precise bounding box.
2. **LLM** sees only the prompt text with `{{slot_id}}` placeholders.
   It returns `{slot_id: answer}`. No coordinates, no rendering.
3. **Renderer** (`render.py`) places each answer in its precomputed bbox
   with auto-fit font sizing.

## Setup

```bash
cd app
pip install flask openai pymupdf
export OPENAI_API_KEY=sk-...
python app.py
```

Open http://localhost:5000

## Project structure

```
app/
├── app.py              Flask routes
├── preprocess.py       PDF → structure JSON
├── render.py           structure JSON + answers → filled PDF
├── templates/
│   └── index.html      Frontend (vanilla HTML/CSS/JS)
├── uploads/            User uploads (one .pdf per job)
└── outputs/            Filled PDFs + preview images
```

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/upload` | POST | multipart PDF → returns `{job_id, units, slot_count, ...}` |
| `/api/fill` | POST | `{job_id}` → calls OpenAI, renders, returns `{answers}` |
| `/api/download/<job_id>` | GET | stream the filled PDF |
| `/api/preview/<job_id>/<page\|filled>/<n>` | GET | PNG preview of original or filled page |

## Config

- `OPENAI_API_KEY` — required, read at process start.
- Model is `gpt-5-mini` (set in `app.py:call_openai_to_fill`). Swap to
  `gpt-5` for tougher worksheets if cost isn't an issue.
- Max upload size is 10 MB (set in `app.py`).

## Known limits

- In-memory job store — restart drops everything. Swap in Redis for
  anything beyond local use.
- No auth — anyone with the URL can use it. Add a key check or rate
  limit before deploying.
- No AcroForm fast-path yet — works fine but slower than necessary for
  fillable government-form-style PDFs.
- The "name/date" header on worksheets sometimes gets generic answers
  ("Student", today's date) since the LLM doesn't know who the user is.
  Eventually you'd let the user pre-fill those.
