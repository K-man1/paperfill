# PaperFill

Paperfill is an AI PDF filler/editor where you upload your PDF and we find the blanks, and then use AI to fill them in. It is a substitute to the terrible coordinate based system that Claude and other LLMs use when editing PDFs, ensuring that no text is in random places.

## How it works

1. It starts with a detector which finds every single possible answer slot. Then it ouputs a JSON structure where each slot has a precise bounding box. Then, it goes through one of two paths (the user chooses which one they wanna do):
   - `preprocess.py`: The user chooses what kind of blanks must be filled before submitting the PDF. Then its just a deterministic PyMuPDF walk. 
   ![Supports fill-in-the-blanks, open responses, bulleted lists, and tables](<Screenshot 2026-06-20 at 10.48.52 AM.png>)
   - `multimodal_preprocess.py`: So in this version, AI looks at the PDF and determines the the questions and the answers. Then the answers anchor to the appropriate boxes and the boxes that are not used are removed. This is also the path for scanned or image-only PDFs (`vision_preprocess.py` does the page-image detection).
2. **LLM** sees only prompt text with `{{slot_id}}` placeholders and returns `{slot_id: answer}` (in the first path, the 2nd path skips this)
3. **Renderer** (`render.py`) places each answer in its bounding box with auto-fit sizing. The user can edit the font, size and location of the box.

You can also give it extra context or instuctions (so far just text, files, and youtube videos).

## Features
- Editing
  - I knew it wasnt gunna be perfect ALL the time, so I added the ability to add, edit, and remove text boxes
  - You can also edit text boxes with AI. Just select a text box and choose lengthen, shorten, or give it another prompt
- AI screenshots
  - So I also knew if ur getting AI to fill the PDF for you, sometimes you may not know the info. that means you can take a "screenshot" of the question and AI will answer it and you can move the answer to the correct location.
- Handwriting
   - Simply upload a filled template of your handwriting, and then you can have PDFs filled in with what looks like your handwriting. (https://github.com/yashlamba/handwrite)
   - <img width="830" height="150" alt="image" src="https://github.com/user-attachments/assets/68de08c8-c446-459e-9073-afa9a497d9e3" />


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
source venv/bin/activate
pip install -r requirements.txt

```

### Configure .env

| Variable | Default | Purpose |
|----------|---------|---------|
| `AI_API_KEY` | bruh | AI key. I use HCAI for a free AI API key for teens. |
| `AI_BASE_URL` / `AI_MODEL` | HCAI proxy / `openai/gpt-5.5` | uhhhhh i think u can guess |
| `EMAIL_AUTH` | `0` (off) | do 1 to enable email/password sign-in alongside Google. |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | — | config for "Sign in with Google". |
| `ADMIN_EMAILS` | `my email` | the emails u wanna have users as admins. they can see the admin dashboard |
| `SUPABASE_URL` / `SUPABASE_SECRET_KEY` | ummm | stores info for admin dashboard |
