"""
OpenAI-compatible client with provider fallback.

Primary is whatever provider AI_API_KEY / AI_BASE_URL point at (the Hack Club
AI proxy by default). If a call
to the primary raises (auth failure, outage, unsupported model, …) it is
retried once against a fallback provider — OpenRouter by default
(OPENROUTER_API_KEY / OPENROUTER_BASE_URL). The fallback uses its own model id
(OPENROUTER_MODEL) since the providers don't share model names.

The wrapper mimics just the surface the app uses — `client.chat.completions
.create(...)` and `client.files.create(...)` — so it drops in wherever a raw
OpenAI client was used, with no call-site changes.

Every chat call is also *metered* here. This is the one place all seven call
sites funnel through, so token counts, latency, which provider actually served
the request, and the estimated cost land in one table without touching any
call site. Wrap a call in `call_context("fill", job_id=…)` to label what the
tokens were spent on; unlabelled calls still record, just as "unknown".
"""

import contextvars
import os
import queue
import threading
import time

# What the current call is for. A ContextVar rather than a global because
# gunicorn's sync workers handle one request per thread and these must not
# bleed between them.
_purpose = contextvars.ContextVar("llm_purpose", default="unknown")
_job_id = contextvars.ContextVar("llm_job_id", default="")
_is_pro = contextvars.ContextVar("llm_is_pro", default=None)
_user_key = contextvars.ContextVar("llm_user_key", default="")


class call_context:
    """Label the AI calls made inside this block, for the spend dashboard.

        with call_context("vision_fill", job_id=job_id, is_pro=True):
            client.chat.completions.create(...)

    `user_key` additionally spends the call's tokens against that account's
    Free-tier daily credit budget (see usage.py) — Pro accounts pass
    is_pro=True and are never metered there.

    Restores the previous labels on exit, so nesting is safe.
    """

    def __init__(self, purpose: str, job_id: str = "", is_pro=None,
                user_key: str = ""):
        self._new = (purpose, job_id, is_pro, user_key)
        self._tokens = None

    def __enter__(self):
        p, j, pro, uk = self._new
        self._tokens = (_purpose.set(p), _job_id.set(j or ""),
                        _is_pro.set(pro), _user_key.set(uk or ""))
        return self

    def __exit__(self, *exc):
        tp, tj, tpro, tuk = self._tokens
        _purpose.reset(tp)
        _job_id.reset(tj)
        _is_pro.reset(tpro)
        _user_key.reset(tuk)
        return False


# ---- Async metering ------------------------------------------------------
# Writing a row to Supabase takes an HTTP round-trip. Doing that inline would
# add that latency to a user's fill, so rows go on a queue and a single daemon
# thread drains it. Bounded so a DB outage can't grow the queue without limit —
# telemetry is the first thing that should be dropped under pressure.

_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=500)
_worker_started = False
_worker_lock = threading.Lock()


def _drain():
    import db
    while True:
        row = _QUEUE.get()
        try:
            db.record_ai_call(row)
        except Exception as e:                       # never die on a bad row
            print(f"[llm] metering write failed: {e}")
        finally:
            _QUEUE.task_done()


def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        threading.Thread(target=_drain, daemon=True,
                         name="llm-metering").start()
        _worker_started = True


def _meter(row: dict) -> None:
    """Hand a finished call to the writer thread. Drops the row instead of
    blocking if the queue is full."""
    try:
        _ensure_worker()
        _QUEUE.put_nowait(row)
    except queue.Full:
        print("[llm] metering queue full — dropping a usage row")
    except Exception as e:
        print(f"[llm] metering skipped: {e}")


def _usage_from(resp) -> tuple[int, int, int]:
    """Pull (prompt, output, total) tokens off a response. Providers that omit
    `usage` (or stream) give zeros rather than an exception."""
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0, 0
    def n(*names):
        for name in names:
            v = getattr(u, name, None)
            if isinstance(v, int):
                return v
        return 0
    p = n("prompt_tokens", "input_tokens")
    o = n("completion_tokens", "output_tokens")
    return p, o, n("total_tokens") or (p + o)


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
        # Only chat completions carry token usage; file uploads pass straight
        # through unmetered.
        meter = self._attr_path == "chat.completions.create"
        t0 = time.monotonic()
        try:
            resp = self._resolve(self._parent.primary)(**kwargs)
            if meter:
                self._record(kwargs.get("model", ""), "primary", t0, resp, None)
            return resp
        except Exception as primary_err:
            fb = self._parent.fallback
            if fb is None:
                if meter:
                    self._record(kwargs.get("model", ""), "primary", t0, None,
                                 primary_err)
                raise
            print(f"[llm] primary provider failed ({type(primary_err).__name__}: "
                  f"{str(primary_err)[:120]}); falling back to "
                  f"{self._parent.fallback_label}")
            # Record the failed primary attempt too: a rising primary-failure
            # rate is exactly what pushes spend onto the paid fallback, and the
            # dashboard can't show that if the failure isn't logged.
            if meter:
                self._record(kwargs.get("model", ""), "primary", t0, None,
                             primary_err)
            fb_kwargs = dict(kwargs)
            # Swap in the fallback's model id when one is configured — the
            # primary's model name usually doesn't exist on the fallback.
            if self._parent.fallback_model and "model" in fb_kwargs:
                fb_kwargs["model"] = self._parent.fallback_model
            t1 = time.monotonic()
            try:
                resp = self._resolve(fb)(**fb_kwargs)
            except Exception as fb_err:
                if meter:
                    self._record(fb_kwargs.get("model", ""), "fallback", t1,
                                 None, fb_err)
                raise
            if meter:
                self._record(fb_kwargs.get("model", ""), "fallback", t1, resp,
                             None)
            return resp

    def _record(self, model: str, provider: str, t0: float, resp, err) -> None:
        """Queue one metering row. Wrapped so telemetry can never break a call
        that otherwise succeeded."""
        try:
            import costs
            p, o, total = _usage_from(resp) if resp is not None else (0, 0, 0)
            uk = _user_key.get()
            if uk and not _is_pro.get() and total > 0:
                # Synchronous and local (a flock'd JSON file, not the DB queue
                # below) so the spend is reflected before this request returns
                # — a burst of parallel calls from one job must not all read
                # the same stale "credits remaining" and overspend.
                try:
                    import usage
                    usage.consume_tokens(uk, total)
                except Exception as e:
                    print(f"[llm] credit consumption failed: {e}")
            _meter({
                "purpose": _purpose.get(),
                "model": model or "",
                "provider": provider,
                "prompt_tokens": p,
                "output_tokens": o,
                "total_tokens": total,
                "cost_usd": costs.estimate(model, provider, p, o),
                "latency_ms": int((time.monotonic() - t0) * 1000),
                "ok": err is None,
                "error": (f"{type(err).__name__}: {str(err)[:200]}"
                          if err is not None else None),
                "is_pro": _is_pro.get(),
                "job_id": _job_id.get() or None,
            })
        except Exception as e:
            print(f"[llm] could not build metering row: {e}")


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
    primary_key = (os.environ.get("AI_API_KEY")
                   or os.environ.get("HCAI_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"))
    if not primary_key:
        raise RuntimeError(
            "No API key found. Set AI_API_KEY in .env or environment."
        )
    primary = _make(
        primary_key,
        (os.environ.get("AI_BASE_URL")
         or os.environ.get("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1")),
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
