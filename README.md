# PaperFill

PaperFill is an AI PDF filler. All you have to do is upload your PDF and then give any additional information you may want. Then, all avaiable blanks (open-space, underscores, tables, etc) will be filled, and you can edit it later!

## How it works

1. It starts with a detector which finds every single possible answer slot. Then it ouputs a JSON structure where each slot has a precise bounding box. After that, it goes through one of two paths (the user chooses which one they wanna do):
   - `preprocess.py` — The user chooses what kind of blanks must be filled before submitting the PDF. Then its just a deterministic PyMuPDF walk. 
   ![Supports fill-in-the-blanks, open responses, bulleted lists, and tables](<Screenshot 2026-06-20 at 10.48.52 AM.png>)
   - `multimodal_preprocess.py` — So in this version, AI looks at the PDF and determines the the questions and the answers. Then the answers anchor to the appropriate boxes and the boxes that are not used are removed. This is also the path for scanned or image-only PDFs (`vision_preprocess.py` does the page-image detection).
2. **LLM** sees only prompt text with `{{slot_id}}` placeholders and returns `{slot_id: answer}` (in the first path, the 2nd path skips this)
3. **Renderer** (`render.py`) places each answer in its bounding box with auto-fit sizing. The user can edit the font, size and location of the box.

You can also feed extra context (a pasted file or a YouTube transcript via `context_sources.py`) so answers are grounded in your material.

## Run it on your own device
### Prerequisites
- Python 3.10+
- `potrace` system binary (only needed for the handwriting-font feature)
  - macOS: `brew install potrace`
  - Debian/Ubuntu: `apt-get install potrace`

### Setup

```bash
git clone https://github.com/K-man1/paperfill
cd PaperFill

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### Configure .env

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_API_KEY` | — | AI key. I use HCAI for a free AI API key for teens. |
| `AI_BASE_URL` / `AI_MODEL` | HCAI proxy / `openai/gpt-5.5` | uhhhhh i think u can guess |
| `EMAIL_AUTH` | `0` (off) | do 1 to enable email/password sign-in alongside Google. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | — | config for "Sign in with Google". |
| `ADMIN_EMAILS` | `my email` | the emails u wanna have users as admins. they can see the admin dashboard |
| `SUPABASE_URL` / `SUPABASE_SECRET_KEY` | — | stores info for admin dashboard |

### Start it
Development:

```bash
source venv/bin/activate
python app.py
```

Open http://localhost:8080

Production (gunicorn):

```bash
./run.sh          # reads .env, activates venv/.venv, runs gunicorn
```

## Project structure

```
app.py                    Flask routes + LLM call
preprocess.py             PDF → structure JSON (deterministic detector)
multimodal_preprocess.py  Vision + anchor-bridge detector
vision_preprocess.py      Page-image slot detection
render.py                 structure JSON + answers → filled PDF
context_sources.py        Extra context (files, YouTube transcripts)
db.py                     Supabase-backed stats / user store
handwriting/              "Your handwriting" font builder (needs potrace)
templates/                Frontend (index.html, handwriting.html, login, admin)
uploads/                  User uploads (one .pdf per job)
outputs/                  Filled PDFs + preview images
```