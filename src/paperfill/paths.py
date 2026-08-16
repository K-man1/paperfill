"""Repo-root anchor.

Every module used to find its data files with ``Path(__file__).parent``, which
worked only while all the code sat in the repo root. Under ``src/paperfill/``
that idiom silently points into the package directory, so runtime files
(uploads, ai_rates.json, the watermark) would be created or looked for in the
wrong place with no import error to warn you.

Anchor off this one constant instead: it resolves to the repo root no matter
how deep the importing module lives.
"""

from pathlib import Path

# paths.py -> parents[0]=paperfill, [1]=src, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
