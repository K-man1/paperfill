# PaperFill
Paperfill is an AI PDF filler which finds the blanks that are in your PDF (open space, fill-in-the-blanks, tables, etc) and uses its knowledge to fill it in. It also has handwriting support, so you can just upload a template of your handwriting and it copies it. 

It's unique from Claude and other LLM's PDF editing tools and other AI PDF editors because it doesnt use a coordinate based system making sure all the text is in the correct spot while Claude often messes up and the words are in the middle of the blank. 


Paperfill: <img width="645" height="78" alt="paperfill" src="https://github.com/user-attachments/assets/9e71ad5c-9ab8-4b30-a299-618af4ee6c61" />

Claude: <img width="628" height="78" alt="Screenshot 2026-07-08 at 6 43 28 PM" src="https://github.com/user-attachments/assets/85ee9e69-56e8-4269-9d79-e1f1cd08fafc" />

I personally use it for my HW (dont tell me teachers 😭) and for quickly completing the low effort clearly-created-with-ChatGPT worksheets that my spanish teacher assigns.

Use it here: [paperfill.hackclub.app](paperfill.hackclub.app)

See the demo: https://www.youtube.com/watch?v=bDNsQvA0_DU plz dont judge the cough

## Features

- Editing
  - I knew it wasnt gunna be perfect ALL the time, so I added the ability to add, edit, and remove text boxes
  - You can also edit text boxes with AI. Just select a text box and choose lengthen, shorten, or give it another prompt
- AI screenshots
  - So I also knew if ur getting AI to fill the PDF for you, sometimes you may not know the info. that means you can take a "screenshot" of the question and AI will answer it and you can move the answer to the correct location.
- Handwriting
  - Simply upload a filled template of your handwriting, and then you can have PDFs filled in with what looks like your handwriting.(https://github.com/yashlamba/handwrite)

## Prerequisites

Python 3.10+

potrace system binary (only needed for the handwriting-font feature)

## Setup
``` 
git clone https://github.com/K-man1/paperfill
cd PaperFill

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run it:

```
./deploy/run.sh                 # production (gunicorn)
python -m pytest                # tests
```

## Layout

```
src/paperfill/        all app code
├── app.py            Flask entrypoint
├── paths.py          REPO_ROOT anchor (see note below)
├── ai/               preprocess, multimodal, vision, llm_client, render
├── data/             db, usage, stats, costs
├── utils/            json_utils, context_sources
└── handwriting/      handwriting-font pipeline
    └── data/         printable template + calibrated geometry
templates/  static/   Flask convention, stay at the repo root
tests/                pytest suite
tools/                dev harnesses (ab_compare.py)
deploy/               Caddyfile, Dockerfile, Procfile, gunicorn.conf.py, run.sh
docs/                 ab-comparison/ — evidence for the accuracy claim
assets/               images the app itself renders (page watermark)
verification/         domain-ownership tokens (see note below)
```

Three things worth knowing before you move files around:

**`src/paperfill/paths.py`.** Modules must not locate *runtime* files with
`Path(__file__).parent` — under `src/paperfill/` that points into the package,
not the repo root, so `uploads/`, `ai_rates.json` and friends would silently be
read from or created in the wrong place with no import error to warn you.
Anchor those off `REPO_ROOT` instead. Same reason `app.py` passes an explicit
`template_folder`/`static_folder` to `Flask()`.

The opposite rule applies to files that **ship with the code**, like
`handwriting/data/template.pdf`: those are package data and *should* be found
relative to `__file__`, because they move when the package moves. The split to
watch is shipped-with-the-code vs created-at-runtime, not old-path vs new-path.
`handwriting/font_store.py` is the one place both appear side by side.

**`verification/`.** These are domain-ownership tokens whose filenames are
fixed by the verifying service, so they must be served from the exact URLs
`/2d7883f358a775fc1a8f.txt` and `/0efb70ed5ecb5409945db6f7bb100589.html`. The
files are kept in `verification/` rather than the repo root, and `app.py` has
two small routes that serve them at those paths.

**`handwriting/fonts/` at the repo root** (gitignored, created on first use)
holds fonts users built from their own handwriting. It deliberately stayed put
when the code moved into `src/`, so existing fonts on a deployed server aren't
orphaned.

Nothing needs `pip install -e .` — `deploy/gunicorn.conf.py` puts `src/` on
`sys.path` and `pyproject.toml` does the same for pytest.
