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
        monkeypatch.setattr("backend.cli._llm_providers", lambda: [llm])
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
