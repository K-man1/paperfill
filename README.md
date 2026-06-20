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
git clone <repo-url> PaperFill
cd PaperFill

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then edit .env (see below)
```

### Configure `.env`

```bash
# generate a session signing key
python -c "import secrets; print(secrets.token_hex(32))"
```

Put that value in `SECRET_KEY`. Useful keys:

| Variable | Default | Purpose |
|----------|---------|---------|
| `SECRET_KEY` | random per-boot | Flask session signing. Set it so sessions survive restarts. |
| `PORT` | `8080` | Port the server binds to. |
| `AI_API_KEY` | — | AI key. I use HCAI for a free AI API key for teens. |
| `AI_BASE_URL` / `AI_MODEL` | proxy defaults / `openai/gpt-5.5` | Point at OpenAI or another provider instead. |
| `PAPERFILL_DETECTOR` | `deterministic` | Set `multimodal` to route uploads through the vision path. Also selectable per-upload. |
| `EMAIL_AUTH` | `0` (off) | Enable email/password sign-in alongside Google. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | — | Enable "Sign in with Google". |
| `ADMIN_EMAILS` | `my email` | Comma-separated email addresses for admins account so that I can see the `/admin` dashboard. |
| `SUPABASE_URL` / `SUPABASE_SECRET_KEY` | — | Backs the admin dashboard stats. |

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

## Key routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Main worksheet-filler UI |
| `/handwriting` | GET | Handwriting-font onboarding |
| `/login`, `/logout` | — | Auth |
| `/auth/google`, `/auth/google/callback` | — | Google OAuth |
| `/admin` | GET | Admin dashboard (stats) |
| `/api/upload` | POST | multipart PDF → `{job_id, slot_count, ...}` |
| `/api/context` | POST | Attach extra context to a job |
| `/api/fill` | POST | `{job_id}` → LLM fills slots, renders, returns answers |
| `/api/update`, `/api/snip`, `/api/refine` | POST | Edit / re-fill specific answers |
| `/api/fonts`, `/api/fonts/template`, `/api/style` | — | Handwriting-font build & apply |
| `/api/download/<job_id>` | GET | Stream the filled PDF |
| `/api/preview/<job_id>/<which>/<page>` | GET | PNG preview of original/filled page |