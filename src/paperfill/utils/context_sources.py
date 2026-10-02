"""
Extra context sources for the worksheet filler.

Beyond the free-text instructions, a user can attach reference material the
AI should lean on when filling the sheet:

  - Reference files: PDFs, plain-text/markdown/csv, or images. Their textual
    content is extracted (images are OCR'd with the same vision model used for
    scanned worksheets) and folded into the instructions block.
  - YouTube videos: the transcript is fetched and treated as study material.
  - The web: Pro only. The model reads the worksheet, decides what it can't
    answer from memory, and those searches run through the proxy's Exa
    endpoint.

Everything here returns plain strings so the caller can concatenate them into
the single `instructions` string that call_openai_to_fill() already honors.
"""

import base64
import os
import re

import fitz
import requests

from paperfill.data import models
from paperfill.utils.json_utils import json_from_response

# Per-source caps so one giant PDF or a 3-hour lecture transcript can't blow
# out the prompt. Tuned to stay well under the model's context window when
# several sources are attached at once.
MAX_FILE_CHARS = 6000
MAX_TRANSCRIPT_CHARS = 8000

TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".rtf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
PDF_EXTS = {".pdf"}


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n…[truncated]"


# ---- File extraction -----------------------------------------------------

def _extract_pdf(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        chunks = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(chunks)


def _extract_text(data: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_image(data: bytes, mime: str, user_key: str = "",
                   is_pro: bool = False) -> str:
    """OCR / describe an image with the vision model.

    Goes through llm_client rather than building a raw OpenAI client so these
    tokens are metered like every other call — reference-image OCR was the one
    path whose spend was invisible — and so it inherits the same fallback.
    """
    from paperfill.ai.llm_client import build_client, call_context

    try:
        client = build_client()
    except RuntimeError:
        return "[image attached, but no API key configured to read it]"
    data_uri = f"data:{mime};base64," + base64.b64encode(data).decode()
    with call_context("context_image", is_pro=is_pro, user_key=user_key):
        resp = client.chat.completions.create(
            model=models.get("vision"),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Transcribe all text in this image verbatim. If it is a "
                        "diagram or photo with little text, describe what it shows "
                        "in a few sentences. Output only the transcription/description."
                    )},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
        )
    return resp.choices[0].message.content or ""


def extract_file_text(filename: str, data: bytes, user_key: str = "",
                      is_pro: bool = False) -> str:
    """
    Return the textual content of an uploaded reference file, truncated to
    MAX_FILE_CHARS. Unsupported types yield a short marker string rather than
    raising, so one bad attachment never sinks the whole fill request.

    `user_key`/`is_pro` identify whose budget an image OCR call spends against;
    without them the tokens are logged but never debited.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    try:
        if ext in PDF_EXTS:
            text = _extract_pdf(data)
        elif ext in TEXT_EXTS:
            text = _extract_text(data)
        elif ext in IMAGE_EXTS:
            mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"
            text = _extract_image(data, mime, user_key, is_pro)
        else:
            return f"[unsupported file type: {filename}]"
    except Exception as e:  # extraction is best-effort
        return f"[could not read {filename}: {e}]"
    return _truncate(text, MAX_FILE_CHARS)


# ---- YouTube -------------------------------------------------------------

def extract_youtube_id(url: str) -> str | None:
    """
    Pull the 11-character video id out of any common YouTube URL form, or
    return None if this doesn't look like a YouTube link.
    """
    if not url:
        return None
    url = url.strip()
    id_re = r"[A-Za-z0-9_-]{11}"

    # A bare 11-char id (must match the whole string, not just contain one).
    if re.fullmatch(id_re, url):
        return url

    # ?v=<id> on watch URLs, anywhere in the query string.
    m = re.search(rf"[?&]v=({id_re})", url)
    if m:
        return m.group(1)

    # Path-based forms: youtu.be/<id>, /embed/<id>, /shorts/<id>, /v/<id>.
    m = re.search(rf"(?:youtu\.be/|/embed/|/shorts/|/v/)({id_re})", url)
    if m:
        return m.group(1)

    return None


def fetch_youtube_transcript(url: str) -> str:
    """
    Fetch the transcript for a YouTube URL and return it as plain text,
    truncated to MAX_TRANSCRIPT_CHARS. Returns a short marker string (never
    raises) when the id can't be parsed, the library is missing, or no
    transcript is available.
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        return f"[not a recognizable YouTube URL: {url}]"
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return "[youtube-transcript-api not installed; cannot read video]"
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        return f"[no transcript available for {url}: {e}]"
    text = " ".join(seg.get("text", "") for seg in segments)
    text = re.sub(r"\s+", " ", text).strip()
    return _truncate(text, MAX_TRANSCRIPT_CHARS)


# ---- Assembly ------------------------------------------------------------

def assemble_context(sources: list[tuple[str, str]]) -> str:
    """
    Turn a list of (label, text) pairs into one labelled context block ready
    to append to the user's instructions. Empty texts are skipped.
    """
    blocks = []
    for label, text in sources:
        text = (text or "").strip()
        if not text:
            continue
        blocks.append(f"--- {label} ---\n{text}")
    return "\n\n".join(blocks)


# ---- Web search ----------------------------------------------------------
# A Pro-only third source, and the only one the user doesn't assemble by hand:
# the model reads the worksheet, says what it can't answer from memory, and
# those queries go to Exa through the same proxy that serves the chat calls.
# Searching is not free — Exa comes off the same daily budget — so a sheet the
# model can already answer has to cost nothing, which is what the planner's
# empty list is for.

# Enough for a handful of results to survive the trip into the fill prompt
# without crowding out the worksheet itself.
MAX_SEARCH_CHARS = 9000
MAX_QUERIES = 3
RESULTS_PER_QUERY = 4
SNIPPET_CHARS = 1200
SEARCH_TIMEOUT = 25

# How much of the worksheet the planner sees. It only has to recognise what the
# questions are about, and a 50-item sheet's full text is mostly repetition.
MAX_PLAN_CHARS = 4000

# Three short search strings, plus room for a reasoning model to think first.
# Set explicitly because the proxy reserves a model's full output window against
# the day's remaining allowance otherwise: unset, this call is refused with a
# 402 whenever the budget runs low and fails over to the paid provider.
PLAN_MAX_TOKENS = 600

_PLAN_SYSTEM = (
    "You are deciding what to look up on the web before a worksheet is "
    "answered. You get the worksheet's questions and any material the user "
    "attached. The user has explicitly asked for a web search, so search "
    "whenever a question turns on a fact that could have changed or that you "
    "would otherwise be guessing at: current events, recent figures, "
    "this year's rankings or prices, a named local detail, a source you would "
    "have to reconstruct from memory.\n"
    "Return NOTHING for what you already know cold or what the attached "
    "material already covers — arithmetic, grammar, vocabulary, a reading "
    "passage, settled history or science.\n"
    "Write each query the way you would type it into a search engine: "
    "specific, standalone, no question marks needed. At most "
    f"{MAX_QUERIES}, and one good query beats three overlapping ones.\n"
    'Return ONLY a JSON object: {"queries": ["...", "..."]}, with an empty '
    "list when the sheet needs no looking up. /no_think"
)


def _api_key() -> str:
    return (os.environ.get("AI_API_KEY") or os.environ.get("HCAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY") or "")


def exa_search(query: str, num_results: int = RESULTS_PER_QUERY) -> list[dict]:
    """
    One Exa search through the AI proxy. Returns
    [{title, url, published, text}], newest-first as Exa ranked them.

    Raises on a refused or unreadable response rather than returning nothing:
    "the search found no pages" and "the search never ran" lead to different
    answers on the sheet, and the caller logs which one happened.
    """
    base = (os.environ.get("AI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL",
                              "https://ai.hackclub.com/proxy/v1"))
    resp = requests.post(
        base.rstrip("/") + "/exa/search",
        headers={"Authorization": f"Bearer {_api_key()}"},
        json={"query": query, "numResults": num_results, "type": "auto",
              "contents": {"text": {"maxCharacters": SNIPPET_CHARS}}},
        timeout=SEARCH_TIMEOUT,
    )
    if resp.status_code >= 400:
        raise ValueError(f"Exa HTTP {resp.status_code}: {resp.text[:200]}")
    payload = resp.json()
    # Exa bills per search against the same daily allowance as the chat calls,
    # and unlike them it never reaches the spend dashboard (that table is keyed
    # on token counts). The log is the only place this shows up.
    cost = (payload.get("costDollars") or {}).get("total")
    print(f"[search] {query!r} -> {len(payload.get('results') or [])} results"
          + (f", ${cost}" if cost is not None else ""))

    results = []
    for r in payload.get("results") or []:
        text = _truncate(r.get("text") or "", SNIPPET_CHARS)
        if not text:
            continue
        results.append({
            "title": (r.get("title") or r.get("url") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "published": (r.get("publishedDate") or "")[:10],
            "text": text,
        })
    return results


def plan_search_queries(questions: list[str], instructions: str = "",
                        job_id: str = "") -> list[str]:
    """
    Ask the model which searches, if any, would help answer this worksheet.
    At most MAX_QUERIES, and an empty list whenever the sheet is self-contained.
    """
    from paperfill.ai.llm_client import build_client, call_context

    sheet = _truncate("\n".join(q for q in questions if q), MAX_PLAN_CHARS)
    if not sheet:
        return []
    user = sheet if not instructions.strip() else (
        f"Material the user already attached:\n"
        f"{_truncate(instructions, MAX_PLAN_CHARS)}\n\n"
        f"Worksheet questions:\n{sheet}"
    )

    # Pro-only feature, so the Pro slot is the only one it can run on.
    with call_context("web_search", job_id=job_id, is_pro=True):
        resp = build_client().chat.completions.create(
            model=models.get("text_fill_pro"),
            messages=[{"role": "system", "content": _PLAN_SYSTEM},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            max_tokens=PLAN_MAX_TOKENS,
        )
    # json_from_response rather than a bare parse: a reasoning model that spends
    # PLAN_MAX_TOKENS thinking returns truncated JSON, and "the reply was cut
    # off" has to reach the log instead of looking like "nothing to search for".
    queries = json_from_response(resp).get("queries")
    if not isinstance(queries, list):
        return []
    clean = []
    for q in queries:
        q = str(q).strip()[:200]
        if q and q not in clean:
            clean.append(q)
    return clean[:MAX_QUERIES]


def web_search_context(questions: list[str], instructions: str = "",
                       job_id: str = "") -> tuple[str, list[str]]:
    """
    Search the web for whatever this worksheet needs. Returns the labelled
    context block and the queries that actually returned something — both empty
    when the model decided the sheet needs no searching.

    Best-effort like the other sources: a planner or search failure leaves the
    fill to run on the model's own knowledge instead of sinking it.
    """
    try:
        queries = plan_search_queries(questions, instructions, job_id)
    except Exception as e:
        print(f"[search] could not plan queries: {e}")
        return "", []

    sources, searched = [], []
    for query in queries:
        try:
            results = exa_search(query)
        except (requests.RequestException, ValueError) as e:
            print(f"[search] {query!r} failed: {e}")
            continue
        if not results:
            continue
        body = "\n\n".join(
            f"{r['title']}" + (f" ({r['published']})" if r["published"] else "")
            + f"\n{r['url']}\n{r['text']}"
            for r in results
        )
        sources.append((f"Web search: {query}", body))
        searched.append(query)
    return _truncate(assemble_context(sources), MAX_SEARCH_CHARS), searched
