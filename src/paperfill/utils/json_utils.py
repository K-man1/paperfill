"""Shared helpers for pulling a JSON object out of an LLM response."""

import json
import re


def json_from_response(resp) -> dict:
    """The JSON object a chat completion returned, or raise saying why not.

    Truncation is the failure this exists for. A response that hits the output
    token cap comes back as an unclosed object, extract_json_object cannot parse
    it, and the {} it falls back to is indistinguishable from a worksheet that
    genuinely has no blanks — a 40-item sheet reported unit_count 0 with nothing
    logged anywhere. Detection callers use this instead of parsing the content
    themselves so an empty result can only mean "the model found nothing".
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise RuntimeError("model returned no choices")
    choice = choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise RuntimeError("model response was truncated at the output token "
                           "cap; the JSON it returned is incomplete")
    obj = extract_json_object(choice.message.content or "")
    if not obj:
        raise RuntimeError("model response contained no JSON object")
    return obj


def extract_json_object(text: str) -> dict:
    """
    Robustly pull a JSON object out of a model response. Handles:
      - <think>...</think> blocks (qwen3 reasoning models)
      - ```json fenced code blocks
      - leading/trailing prose
    Falls back to the first {...} balanced span if direct parse fails.
    """
    if not text:
        return {}
    # Strip reasoning blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Last resort: find the first balanced { ... } and try that
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    break
    return {}
