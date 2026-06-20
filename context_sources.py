"""
Extra context sources for the worksheet filler.

Beyond the free-text instructions, a user can attach reference material the
AI should lean on when filling the sheet:

  - Reference files: PDFs, plain-text/markdown/csv, or images. Their textual
    content is extracted (images are OCR'd with the same vision model used for
    scanned worksheets) and folded into the instructions block.
  - YouTube videos: the transcript is fetched and treated as study material.

Everything here returns plain strings so the caller can concatenate them into
the single `instructions` string that call_openai_to_fill() already honors.
"""

import base64
import os
import re

import fitz

# Per-source caps so one giant PDF or a 3-hour lecture transcript can't blow
# out the prompt. Tuned to stay well under the model's context window when
# several sources are attached at once.
MAX_FILE_CHARS = 6000
MAX_TRANSCRIPT_CHARS = 8000

TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".rtf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
PDF_EXTS = {".pdf"}

VISION_MODEL = os.environ.get("VISION_MODEL", "openai/gpt-5.5")


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


def _extract_image(data: bytes, mime: str) -> str:
    """OCR / describe an image with the vision model (reuses the proxy client)."""
    from openai import OpenAI

    api_key = (os.environ.get("AI_API_KEY")
               or os.environ.get("HCAI_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        return "[image attached, but no API key configured to read it]"
    client = OpenAI(
        api_key=api_key,
        base_url=(os.environ.get("AI_BASE_URL")
                  or os.environ.get("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1")),
    )
    data_uri = f"data:{mime};base64," + base64.b64encode(data).decode()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
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


def extract_file_text(filename: str, data: bytes) -> str:
    """
    Return the textual content of an uploaded reference file, truncated to
    MAX_FILE_CHARS. Unsupported types yield a short marker string rather than
    raising, so one bad attachment never sinks the whole fill request.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    try:
        if ext in PDF_EXTS:
            text = _extract_pdf(data)
        elif ext in TEXT_EXTS:
            text = _extract_text(data)
        elif ext in IMAGE_EXTS:
            mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else f"image/{ext.lstrip('.')}"
            text = _extract_image(data, mime)
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
