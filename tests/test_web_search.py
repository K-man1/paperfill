"""
Tests for the Pro-only web search.

Two things decide whether this is right. Searching costs money off a shared
daily allowance, so a worksheet the model can already answer must not trigger
one, and a search that fails must leave the fill running rather than sinking
it. The rest — that a free account can't reach it at all — lives in
test_pro_access.py alongside the other tier gates.

Neither the planner nor Exa is real here: the planner is a stub chat client and
Exa is a stub requests.post, so nothing reaches the network.
"""

import json
import types

import pytest
import requests

from paperfill.utils import context_sources as cs


QUESTIONS = ["Who won the 2026 World Cup?", "Name the host cities."]


def fake_completion(content: str, finish_reason: str = "stop"):
    """The attribute shape plan_search_queries reads off a response."""
    message = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[
        types.SimpleNamespace(message=message, finish_reason=finish_reason)])


@pytest.fixture
def planner(monkeypatch):
    """Stub the chat client. Set `planner.reply` to what the model returns;
    read `planner.prompts` for what it was asked."""
    box = types.SimpleNamespace(reply='{"queries": []}', finish_reason="stop",
                                prompts=[], kwargs=[])

    def create(**kwargs):
        box.prompts.append(kwargs["messages"][-1]["content"])
        box.kwargs.append(kwargs)
        return fake_completion(box.reply, box.finish_reason)

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)))
    monkeypatch.setattr("paperfill.ai.llm_client.build_client", lambda: client)
    monkeypatch.setattr(cs.models, "get", lambda slot: "test/model")
    return box


@pytest.fixture
def exa(monkeypatch):
    """Stub the Exa endpoint. Set `exa.results` (or `exa.error`); read
    `exa.queries` for what was actually searched."""
    box = types.SimpleNamespace(results=[], error=None, status=200, queries=[])

    def post(url, headers=None, json=None, timeout=None):
        assert url.endswith("/exa/search")
        box.queries.append(json["query"])
        if box.error:
            raise box.error
        return types.SimpleNamespace(
            status_code=box.status,
            text="upstream said no",
            json=lambda: {"results": box.results, "costDollars": {"total": 0.005}},
        )

    monkeypatch.setattr(cs.requests, "post", post)
    return box


def result(title, url, text, published=""):
    return {"title": title, "url": url, "text": text, "publishedDate": published}


# ---- Only search when the sheet actually needs it ------------------------

def test_a_self_contained_sheet_costs_no_searches(planner, exa):
    """The planner's empty list is the whole point: a multiplication drill must
    not spend the daily Exa allowance."""
    planner.reply = '{"queries": []}'
    context, searched = cs.web_search_context(["What is 7 x 8?"])
    assert (context, searched) == ("", [])
    assert exa.queries == []


def test_queries_the_planner_asks_for_are_searched(planner, exa):
    planner.reply = '{"queries": ["2026 world cup winner"]}'
    exa.results = [result("Final result", "https://example.com/final",
                          "Argentina won 2-1.", "2026-07-19T00:00:00.000Z")]

    context, searched = cs.web_search_context(QUESTIONS)
    assert exa.queries == ["2026 world cup winner"]
    assert searched == ["2026 world cup winner"]
    assert "Argentina won 2-1." in context
    assert "https://example.com/final" in context
    assert "2026-07-19" in context      # date only, not the full timestamp


def test_the_planner_is_capped_and_deduped(planner, exa):
    """A model that returns ten near-identical queries would spend ten searches
    on one question."""
    planner.reply = json.dumps({"queries": ["a", "a", "b", "c", "d", "e"]})
    exa.results = [result("t", "https://e.com", "body")]

    _, searched = cs.web_search_context(QUESTIONS)
    assert len(searched) == cs.MAX_QUERIES
    assert len(set(searched)) == len(searched)
    assert exa.queries == searched


def test_attached_material_reaches_the_planner(planner, exa):
    """It has to see what the user already attached, or it searches the web for
    what is sitting in the reference PDF."""
    cs.web_search_context(QUESTIONS, instructions="Answer key: Argentina won.")
    assert "Answer key: Argentina won." in planner.prompts[0]


# ---- A failed search must not cost the user their fill -------------------

def test_a_planner_failure_leaves_the_fill_to_run(planner, exa, monkeypatch):
    def boom():
        raise RuntimeError("model is down")
    monkeypatch.setattr("paperfill.ai.llm_client.build_client", boom)

    assert cs.web_search_context(QUESTIONS) == ("", [])
    assert exa.queries == []


def test_a_search_that_errors_is_skipped_not_raised(planner, exa):
    planner.reply = '{"queries": ["one", "two"]}'
    exa.error = requests.ConnectionError("exa unreachable")

    assert cs.web_search_context(QUESTIONS) == ("", [])


def test_an_http_error_from_exa_is_skipped(planner, exa):
    planner.reply = '{"queries": ["one"]}'
    exa.status = 429

    assert cs.web_search_context(QUESTIONS) == ("", [])


def test_one_failing_query_does_not_lose_the_others(planner, exa, monkeypatch):
    planner.reply = '{"queries": ["good", "bad"]}'
    real_results = [result("t", "https://e.com", "usable body")]

    def post(url, headers=None, json=None, timeout=None):
        if json["query"] == "bad":
            raise requests.Timeout("slow")
        return types.SimpleNamespace(
            status_code=200, text="",
            json=lambda: {"results": real_results})

    monkeypatch.setattr(cs.requests, "post", post)
    context, searched = cs.web_search_context(QUESTIONS)
    assert searched == ["good"]
    assert "usable body" in context


def test_a_query_that_finds_nothing_is_not_reported_as_searched(planner, exa):
    """`searched` is what the UI tells the user it looked up, so a query that
    came back empty doesn't belong in it."""
    planner.reply = '{"queries": ["obscure thing"]}'
    exa.results = []

    context, searched = cs.web_search_context(QUESTIONS)
    assert (context, searched) == ("", [])


def test_a_reply_cut_off_at_the_token_cap_searches_nothing(planner, exa):
    """A reasoning model can spend the whole cap thinking and return half an
    object. That has to read as a failure, not as "nothing to look up"."""
    planner.reply = '{"queries": ["2026 world cup fi'
    planner.finish_reason = "length"

    assert cs.web_search_context(QUESTIONS) == ("", [])
    assert exa.queries == []


def test_the_planner_call_caps_its_own_output(planner, exa):
    """Left unset, the proxy reserves the model's whole output window against
    the day's allowance and 402s the call onto the paid fallback."""
    cs.web_search_context(QUESTIONS)
    assert planner.kwargs[0]["max_tokens"] == cs.PLAN_MAX_TOKENS


def test_a_planner_reply_that_is_not_a_query_list_searches_nothing(planner, exa):
    planner.reply = '{"queries": "just one string"}'
    assert cs.web_search_context(QUESTIONS) == ("", [])
    assert exa.queries == []


# ---- Size -----------------------------------------------------------------

def test_the_block_cannot_crowd_out_the_worksheet(planner, exa):
    """Three searches of four long pages each would otherwise dwarf the sheet
    in the fill prompt."""
    planner.reply = '{"queries": ["a", "b", "c"]}'
    exa.results = [result(f"t{i}", f"https://e.com/{i}", "x" * 5000)
                   for i in range(cs.RESULTS_PER_QUERY)]

    context, _ = cs.web_search_context(QUESTIONS)
    assert len(context) <= cs.MAX_SEARCH_CHARS + 20   # + the truncation marker
