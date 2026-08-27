"""
What each AI call costs — a rate card you configure, not one we guess.

Token *counts* come from the provider's response and are facts. Token *prices*
are not: they change, they differ per provider, and a wrong number here turns
the whole P&L chart into confident fiction. So nothing is hardcoded from
memory. Rates live in ai_rates.json, edited from the admin dashboard, and any
model without a rate is reported as "uncosted" rather than silently counted as
free.

Two things ship with real defaults. The primary provider: PaperFill runs on the
Hack Club AI proxy, which is free, so calls that land on it genuinely cost $0.
And the Gemini models we actually route to on the paid OpenRouter fallback,
whose published rates are in BUILTIN_RATES below. Any other model starts unset —
fill it in from your provider's pricing page.

Prices are USD per 1,000,000 tokens, quoted separately for input and output
because every provider charges more for output.
"""

from __future__ import annotations

import json

from paperfill.paths import REPO_ROOT

_PATH = REPO_ROOT / "ai_rates.json"

# Models the app can reach today, so the admin editor can pre-list them even
# before any call has been made.
def known_models() -> list[str]:
    from paperfill.data import models
    return sorted(set(models.resolved().values()))


# OpenRouter list prices, USD per 1M tokens. These are the fallback models the
# app actually routes to, so shipping them means a fresh deploy costs its calls
# correctly with an empty ai_rates.json. A saved rate for the same model wins.
BUILTIN_RATES = {
    "google/gemini-3-flash-preview": {"in": 0.50, "out": 3.00},
    "google/gemini-3.5-flash": {"in": 1.50, "out": 9.00},
}


def _defaults() -> dict:
    return {
        # The free Hack Club proxy. Genuinely $0 — this is the one rate we can
        # assert without looking it up.
        "_provider_primary_is_free": True,
        "models": dict(BUILTIN_RATES),
    }


def load() -> dict:
    """The saved rate card, or defaults when nothing has been configured."""
    try:
        data = json.loads(_PATH.read_text())
        if not isinstance(data, dict):
            return _defaults()
    except (OSError, json.JSONDecodeError):
        return _defaults()
    base = _defaults()
    saved_models = data.pop("models", None)
    base.update(data)
    if isinstance(saved_models, dict):
        base["models"].update(saved_models)
    return base


def save(models: dict, primary_is_free: bool) -> dict:
    """Persist the rate card. `models` maps model id -> {"in": $/1M, "out": $/1M};
    entries that don't parse as numbers are dropped rather than stored as junk."""
    clean = {}
    for name, rate in (models or {}).items():
        name = str(name).strip()
        if not name:
            continue
        try:
            cin = float((rate or {}).get("in", 0) or 0)
            cout = float((rate or {}).get("out", 0) or 0)
        except (TypeError, ValueError):
            continue
        if cin < 0 or cout < 0:
            continue
        clean[name] = {"in": cin, "out": cout}
    data = {"_provider_primary_is_free": bool(primary_is_free), "models": clean}
    try:
        _PATH.write_text(json.dumps(data, indent=2))
    except OSError as e:
        print(f"[costs] could not save rate card: {e}")
    return data


def estimate(model: str, provider: str, prompt_tokens: int,
             output_tokens: int) -> float | None:
    """Cost in USD for one call, or None when we have no rate to apply.

    None is deliberately different from 0.0: "we don't know" must not look like
    "it was free" on a profit chart. Callers surface uncosted calls separately.
    """
    card = load()
    # A call served by the free proxy costs nothing regardless of model.
    if provider == "primary" and card.get("_provider_primary_is_free", True):
        return 0.0
    rate = (card.get("models") or {}).get(model or "")
    if not rate:
        return None
    try:
        return (float(prompt_tokens or 0) / 1_000_000.0 * float(rate.get("in", 0))
                + float(output_tokens or 0) / 1_000_000.0 * float(rate.get("out", 0)))
    except (TypeError, ValueError):
        return None
