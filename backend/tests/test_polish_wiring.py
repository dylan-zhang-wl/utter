"""P2a Task 8 — polish, from config to the daemon.

Everything below this layer (prompts, levels, the length-ratio guard, the
fall-back-to-raw contract) has been built and tested since P1. Nothing ever
constructed the callable, so `polish_enabled: true` did precisely nothing —
the same illusion as v1's Save button. These tests are about the wire.
"""

import pytest

from backend.cli import build_polish
from backend.config import AppConfig
from backend.providers.llm import LlmError


class FakeLlm:
    id = "gemini"
    display_name = "Fake Gemini"

    def __init__(self, reply="补好标点的文字。", available=True, boom=False):
        self.reply, self._available, self.boom = reply, available, boom
        self.calls = []

    def is_available(self):
        return (True, "") if self._available else (False, "no API key stored")

    def complete(self, system, user):
        self.calls.append({"system": system, "user": user})
        if self.boom:
            raise LlmError("429 rate limited")
        return self.reply


@pytest.fixture
def wired(monkeypatch):
    def install(llm):
        # Signature matters: _llm_providers takes a config now, because Vertex
        # needs a project id to be constructed at all.
        monkeypatch.setattr("backend.cli._llm_providers", lambda config=None: [llm])
        return llm
    return install


def test_polish_off_builds_nothing():
    assert build_polish(AppConfig(polish_enabled=False)) is None


def test_polish_on_builds_a_callable(wired):
    wired(FakeLlm())
    assert build_polish(AppConfig(polish_enabled=True, llm_provider="gemini")) is not None


def test_polish_on_without_a_key_degrades_to_off(wired, capsys):
    """铁律 8. A missing key costs the polish, never the words — and it says so,
    because the author switched this on and expects it to happen."""
    wired(FakeLlm(available=False))
    assert build_polish(AppConfig(polish_enabled=True, llm_provider="gemini")) is None
    assert "润色开着，但" in capsys.readouterr().out


def test_the_level_reaches_the_prompt(wired):
    llm = wired(FakeLlm())
    build_polish(AppConfig(polish_enabled=True, llm_provider="gemini", polish_level="heavy"))(
        "一段口述"
    )
    assert "分段" in llm.calls[0]["system"]


def test_the_vocabulary_reaches_the_prompt(wired):
    """The terminology list is the cheapest correction there is and it applies
    to polish too — a model that has not been told 「foreignisation」 will
    happily 'correct' it."""
    llm = wired(FakeLlm())
    config = AppConfig(polish_enabled=True, llm_provider="gemini", vocabulary=["foreignisation"])
    build_polish(config)("异化 foreignisation")
    assert "foreignisation" in llm.calls[0]["user"]


def test_a_failing_model_returns_the_original_text(wired):
    wired(FakeLlm(boom=True))
    polish = build_polish(AppConfig(polish_enabled=True, llm_provider="gemini"))
    assert polish("原话还在") == "原话还在"


def test_a_model_that_answers_instead_of_editing_is_rejected(wired):
    """"好的，我来帮你处理" is not a polished transcript. The length-ratio guard
    catches it, and the raw text wins."""
    wired(FakeLlm(reply="好的"))
    polish = build_polish(AppConfig(polish_enabled=True, llm_provider="gemini"))
    long = "这是一段完整的学术口述内容，包含若干个分句，需要被补上标点"
    assert polish(long) == long


def test_an_unknown_provider_does_not_crash_dictation(wired):
    wired(FakeLlm())
    assert build_polish(AppConfig(polish_enabled=True, llm_provider="不存在")) is None


# --- model resolution ----------------------------------------------------------


class _Models:
    def __init__(self, ids):
        self._ids = ids

    def list(self):
        return [type("M", (), {"id": i})() for i in self._ids]


class _Client:
    def __init__(self, ids):
        self.models = _Models(ids)


def test_the_model_is_resolved_against_the_key_not_hardcoded(monkeypatch):
    """gpt-4o-mini was this file's default until OpenAI's lineup had moved on
    twice. A hardcoded id is a 404 waiting for a date."""
    from backend.providers import openai_compat as oc

    p = oc.OpenAICompatProvider()
    monkeypatch.setattr(p, "_client", lambda: _Client(["gpt-5.4-mini", "gpt-4o-mini"]))
    monkeypatch.setattr(p, "_key", lambda: "sk-test")

    # Preference order comes from measurement, not from tier names — see the
    # comment on PREFERRED_MODELS.
    assert p.resolve_model() == "gpt-5.4-mini"


def test_an_explicit_model_is_honoured_as_given(monkeypatch):
    from backend.providers import openai_compat as oc

    p = oc.OpenAICompatProvider(model="o9-turbo-imaginary")
    monkeypatch.setattr(p, "_client", lambda: _Client(["gpt-5-nano"]))
    assert p.resolve_model() == "o9-turbo-imaginary"


def test_a_failed_listing_does_not_stop_polish(monkeypatch):
    """铁律 8 again: not knowing the best model must not cost the words."""
    from backend.providers import openai_compat as oc

    p = oc.OpenAICompatProvider()

    def boom():
        raise oc.LlmError("network down")

    monkeypatch.setattr(p, "available_models", boom)
    assert p.resolve_model() == oc.DEFAULT_MODEL


def test_non_chat_models_are_never_offered(monkeypatch):
    """Asking an embedding model to punctuate gives a confusing error rather
    than an obviously wrong answer."""
    from backend.providers import openai_compat as oc

    p = oc.OpenAICompatProvider()
    monkeypatch.setattr(p, "_client", lambda: _Client([
        "gpt-5-nano", "text-embedding-3-small", "whisper-1", "dall-e-3",
        "tts-1", "omni-moderation-latest", "gpt-5-mini",
    ]))
    monkeypatch.setattr(p, "_key", lambda: "sk-test")

    assert p.chat_models() == ["gpt-5-mini", "gpt-5-nano"]


def test_a_model_that_rejects_temperature_zero_is_retried_without_it(monkeypatch):
    """The whole GPT-5 line answers `temperature: 0` with a 400 — "Only the
    default (1) value is supported" — so a hardcoded 0.0 locked this provider
    out of every model newer than 4.1, and the error was being swallowed into a
    bare class name."""
    from backend.providers import openai_compat as oc

    seen = []

    class _Completions:
        def create(self, **kwargs):
            seen.append("temperature" in kwargs)
            if "temperature" in kwargs:
                raise RuntimeError(
                    "Error code: 400 - Unsupported value: 'temperature' does not "
                    "support 0.0 with this model."
                )
            return type("R", (), {"choices": [type("C", (), {
                "message": type("M", (), {"content": "补好标点。"})()
            })()]})()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    p = oc.OpenAICompatProvider(model="gpt-5.4-mini")
    monkeypatch.setattr(p, "_client", lambda **k: _Client())
    monkeypatch.setattr(p, "_key", lambda: "sk-test")

    assert p.complete("sys", "user") == "补好标点。"
    assert seen == [True, False], "tried with, then without"

    seen.clear()
    p.complete("sys", "user")
    assert seen == [False], "and remembered, so it does not pay the 400 twice"


def test_the_servers_error_message_survives(monkeypatch):
    """A 400 explaining exactly what is wrong was arriving as `LlmError` and
    cost a separate investigation to read."""
    from backend.providers import openai_compat as oc

    class _Completions:
        def create(self, **kwargs):
            exc = RuntimeError("boom")
            exc.body = {"error": {"message": "model `gpt-9` does not exist"}}
            raise exc

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    p = oc.OpenAICompatProvider(model="gpt-9")
    monkeypatch.setattr(p, "_client", lambda **k: _Client())
    monkeypatch.setattr(p, "_key", lambda: "sk-test")

    with pytest.raises(oc.LlmError, match="does not exist"):
        p.complete("sys", "user")


def test_the_client_has_a_timeout(monkeypatch):
    """The SDK default is ten minutes. Polish runs on the single worker that
    delivers text in order (铁律 11), so one hung request stops every later
    utterance — worse than an error, and 铁律 8 cannot catch it."""
    from backend.providers import openai_compat as oc

    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    p = oc.OpenAICompatProvider()
    monkeypatch.setattr(p, "_key", lambda: "sk-test")
    p._client()

    assert captured["timeout"] == oc.REQUEST_TIMEOUT
    assert captured["max_retries"] == 1, "2 retries turns a 12s ceiling into 36"
