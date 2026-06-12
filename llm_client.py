"""
OpenAI-compatible client with provider fallback.

Primary is the Hack Club AI proxy (HCAI_API_KEY / OPENAI_BASE_URL). If a call
to the primary raises (auth failure, outage, unsupported model, …) it is
retried once against a fallback provider — OpenRouter by default
(OPENROUTER_API_KEY / OPENROUTER_BASE_URL). The fallback uses its own model id
(OPENROUTER_MODEL) since the providers don't share model names.

The wrapper mimics just the surface the app uses — `client.chat.completions
.create(...)` and `client.files.create(...)` — so it drops in wherever a raw
OpenAI client was used, with no call-site changes.
"""

import os


def _make(api_key: str, base_url: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


class _Method:
    """One bound endpoint (e.g. chat.completions.create) with fallback."""

    def __init__(self, parent: "FallbackClient", attr_path: str):
        self._parent = parent
        self._attr_path = attr_path  # e.g. "chat.completions.create"

    def _resolve(self, client):
        obj = client
        for part in self._attr_path.split("."):
            obj = getattr(obj, part)
        return obj

    def __call__(self, **kwargs):
        try:
            return self._resolve(self._parent.primary)(**kwargs)
        except Exception as primary_err:
            fb = self._parent.fallback
            if fb is None:
                raise
            print(f"[llm] primary provider failed ({type(primary_err).__name__}: "
                  f"{str(primary_err)[:120]}); falling back to "
                  f"{self._parent.fallback_label}")
            fb_kwargs = dict(kwargs)
            # Swap in the fallback's model id when one is configured — the
            # primary's model name usually doesn't exist on the fallback.
            if self._parent.fallback_model and "model" in fb_kwargs:
                fb_kwargs["model"] = self._parent.fallback_model
            return self._resolve(fb)(**fb_kwargs)


class _Namespace:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class FallbackClient:
    def __init__(self, primary, fallback, fallback_model: str | None,
                 fallback_label: str):
        self.primary = primary
        self.fallback = fallback
        self.fallback_model = fallback_model
        self.fallback_label = fallback_label
        # Expose the same attribute paths the OpenAI client does.
        self.chat = _Namespace(
            completions=_Namespace(create=_Method(self, "chat.completions.create"))
        )
        self.files = _Namespace(create=_Method(self, "files.create"))


def build_client() -> FallbackClient:
    """Build the primary (Hack Club) client plus an optional OpenRouter
    fallback. Raises if no primary key is configured."""
    primary_key = os.environ.get("HCAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not primary_key:
        raise RuntimeError(
            "No API key found. Set HCAI_API_KEY in .env or environment."
        )
    primary = _make(
        primary_key,
        os.environ.get("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1"),
    )

    fb_key = os.environ.get("OPENROUTER_API_KEY")
    fallback = None
    fallback_model = None
    if fb_key:
        fallback = _make(
            fb_key,
            os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        fallback_model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")

    return FallbackClient(primary, fallback, fallback_model, "OpenRouter")
