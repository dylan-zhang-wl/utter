"""P1 Task 10 — LLM providers for translation and polish.

The prompt tests are the important ones. 铁律 10 says polish may delete filler,
add punctuation and split paragraphs, and may not reword or change content —
because the author dictates academic argument, and an LLM that turns "翻译不是
复制，而是一种重写" into something more fluent but different is a failure the
author will not catch on re-reading.

That rule can only live in the prompt, so the prompt is under test.
"""

import sys
import types

import pytest

from backend.providers import llm


class FakeLlm:
    def __init__(self, id_="fake", available=True, reason="", reply="polished"):
        self.id = id_
        self.display_name = f"Fake {id_}"
        self._available = available
        self._reason = reason
        self._reply = reply
        self.calls = []

    def is_available(self):
        return (self._available, self._reason)

    def complete(self, system, user):
        self.calls.append({"system": system, "user": user})
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


# --- 铁律 10: the polish prompt ----------------------------------------------


@pytest.mark.parametrize("level", ["light", "medium", "heavy"])
def test_every_polish_level_forbids_changing_content(level):
    prompt = llm.polish_prompt(level)
    assert "不得增删内容" in prompt


@pytest.mark.parametrize("level", ["light", "medium", "heavy"])
def test_every_polish_level_forbids_rewording(level):
    """Even "heavy" only reorganises; rewording is never on the table."""
    assert "不得改写措辞" in llm.polish_prompt(level)


def test_light_is_the_most_conservative():
    light = llm.polish_prompt("light")
    assert "标点" in light
    assert "口水词" not in light, "light only punctuates; removing filler is medium"


def test_light_and_medium_ask_the_model_for_exactly_the_same_thing():
    """They used to differ, and that difference is what cost the author 「因为」:
    「medium」 told the model to delete filler and it deleted a causal
    connective too. Deletion is now ours, done from a fixed list. The levels
    differ in what *we* do afterwards, not in what the model is asked."""
    assert llm.polish_prompt("light") == llm.polish_prompt("medium")


def test_heavy_is_the_only_level_that_asks_for_more():
    assert llm.polish_prompt("heavy") != llm.polish_prompt("light")
    assert "分段" in llm.polish_prompt("heavy")


def test_levels_actually_differ_in_behaviour():
    kept, _ = llm.safe_polish(FakeLlm(reply="呃，我觉得是这样。"), "呃我觉得是这样", level="light")
    dropped, _ = llm.safe_polish(FakeLlm(reply="呃，我觉得是这样。"), "呃我觉得是这样", level="medium")
    assert kept == "呃，我觉得是这样。"
    assert dropped == "我觉得是这样。"


def test_unknown_level_falls_back_to_light():
    """A hand-edited config must degrade to the safe setting, not the risky one."""
    assert llm.polish_prompt("aggressive") == llm.polish_prompt("light")


def test_polish_and_translate_send_different_prompts():
    provider = FakeLlm()
    llm.polish(provider, "some text")
    llm.translate(provider, "some text")

    assert provider.calls[0]["system"] != provider.calls[1]["system"]


# --- context and vocabulary --------------------------------------------------


def test_polish_passes_previous_paragraph_as_context():
    """§4.1b: only the previous paragraph, so request size stays constant
    however long the dictation runs."""
    provider = FakeLlm()
    llm.polish(provider, "second bit", context="first bit")

    assert "first bit" in provider.calls[0]["user"]


def test_polish_without_context_does_not_invent_one():
    provider = FakeLlm()
    llm.polish(provider, "only bit")
    assert "only bit" in provider.calls[0]["user"]


def test_vocabulary_reaches_the_prompt():
    provider = FakeLlm()
    llm.polish(provider, "text", vocabulary=["Venuti", "异化"])

    blob = provider.calls[0]["system"] + provider.calls[0]["user"]
    assert "Venuti" in blob
    assert "异化" in blob


def test_translate_targets_chinese():
    provider = FakeLlm()
    llm.translate(provider, "Translation is rewriting.")
    assert "中文" in provider.calls[0]["system"]


# --- 铁律 8: a polish failure must never cost the transcript ------------------


def test_provider_error_becomes_llm_error():
    provider = FakeLlm(reply=RuntimeError("connection refused"))
    with pytest.raises(llm.LlmError):
        llm.polish(provider, "text")


def test_safe_polish_returns_the_raw_text_on_failure():
    """The caller finalises the utterance either way. Losing a sentence because
    a language model timed out is the failure mode 铁律 8 exists to prevent."""
    provider = FakeLlm(reply=RuntimeError("connection refused"))
    text, polished = llm.safe_polish(provider, "the raw words")

    assert text == "the raw words"
    assert polished is False


def test_safe_polish_reports_success():
    """Punctuation is taken from the model; letters come from the transcript,
    capitalisation included.

    That is deliberate, and it earns its keep on this author's vocabulary.
    「van Leeuwen」 is spelled with a lowercase v when the full name is given,
    and a model asked to tidy English will confidently "correct" it. The full
    stop is a gain; the capital is a guess."""
    text, polished = llm.safe_polish(FakeLlm(reply="The raw words."), "the raw words")
    assert text == "the raw words."
    assert polished is True


def test_safe_polish_with_no_provider_returns_raw():
    text, polished = llm.safe_polish(None, "the raw words")
    assert text == "the raw words"
    assert polished is False


def test_safe_polish_rejects_a_suspiciously_short_result():
    """A model that answers "OK" or drops half the text must not silently
    replace what the user said."""
    text, polished = llm.safe_polish(FakeLlm(reply="OK"), "a much longer sentence here")

    assert text == "a much longer sentence here"
    assert polished is False


def test_safe_polish_rejects_an_empty_result():
    text, polished = llm.safe_polish(FakeLlm(reply="   "), "the raw words")
    assert text == "the raw words"
    assert polished is False


def test_safe_translate_returns_none_on_failure():
    """Design §5: a failed translation must not take the English with it."""
    provider = FakeLlm(reply=RuntimeError("rate limited"))
    assert llm.safe_translate(provider, "some english") is None


# --- Ollama ------------------------------------------------------------------


@pytest.fixture
def fake_httpx(monkeypatch):
    state = {"tags": 200, "chat_reply": "polished text", "requests": []}

    class FakeResponse:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise llm.httpx.HTTPStatusError(
                    "boom", request=None, response=None
                )

    def get(url, **kwargs):
        state["requests"].append(("GET", url))
        if state["tags"] == "unreachable":
            raise llm.httpx.ConnectError("connection refused")
        return FakeResponse(state["tags"], {"models": [{"name": "qwen3:4b"}]})

    def post(url, **kwargs):
        state["requests"].append(("POST", url, kwargs.get("json")))
        return FakeResponse(200, {"message": {"content": state["chat_reply"]}})

    monkeypatch.setattr(llm.httpx, "get", get)
    monkeypatch.setattr(llm.httpx, "post", post)
    return state


def test_ollama_unavailable_when_the_daemon_is_down(fake_httpx):
    from backend.providers.ollama import OllamaProvider

    fake_httpx["tags"] = "unreachable"
    available, reason = OllamaProvider().is_available()

    assert available is False
    assert "ollama" in reason.lower()


def test_ollama_available_when_the_daemon_answers(fake_httpx):
    from backend.providers.ollama import OllamaProvider

    available, _ = OllamaProvider().is_available()
    assert available is True


def test_ollama_completes(fake_httpx):
    from backend.providers.ollama import OllamaProvider

    assert OllamaProvider().complete("sys", "user") == "polished text"


def test_ollama_sends_both_messages(fake_httpx):
    from backend.providers.ollama import OllamaProvider

    OllamaProvider().complete("the system prompt", "the user text")
    payload = fake_httpx["requests"][-1][2]
    roles = {m["role"]: m["content"] for m in payload["messages"]}

    assert roles["system"] == "the system prompt"
    assert roles["user"] == "the user text"


def test_ollama_pins_temperature_low(fake_httpx):
    """Polish must be boring and repeatable. A creative sampler is exactly how
    an LLM starts rewriting arguments."""
    from backend.providers.ollama import OllamaProvider

    OllamaProvider().complete("sys", "user")
    payload = fake_httpx["requests"][-1][2]

    assert payload["options"]["temperature"] == 0.0


# --- OpenAI-compatible -------------------------------------------------------


@pytest.fixture
def fake_openai(monkeypatch):
    state = {"init": [], "calls": [], "reply": "polished text"}

    class FakeCompletions:
        def create(self, **kwargs):
            state["calls"].append(kwargs)
            message = types.SimpleNamespace(content=state["reply"])
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            state["init"].append(kwargs)
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return state


def test_openai_compat_reads_the_key_from_the_keychain(monkeypatch, fake_openai):
    """铁律 4. Never from config JSON — the config file does not even have the
    field any more."""
    from backend.providers import openai_compat

    monkeypatch.setattr(openai_compat.secrets, "get_secret", lambda name: "sk-from-keychain")
    openai_compat.OpenAICompatProvider().complete("sys", "user")

    assert fake_openai["init"][0]["api_key"] == "sk-from-keychain"


def test_openai_compat_unavailable_without_a_key(monkeypatch, fake_openai):
    from backend.providers import openai_compat

    monkeypatch.setattr(openai_compat.secrets, "get_secret", lambda name: None)
    available, reason = openai_compat.OpenAICompatProvider().is_available()

    assert available is False
    assert "key" in reason.lower()


def test_openai_compat_honours_a_custom_base_url(monkeypatch, fake_openai):
    """One provider covers DeepSeek, Groq, 硅基流动 — they are all this API."""
    from backend.providers import openai_compat

    monkeypatch.setattr(openai_compat.secrets, "get_secret", lambda name: "sk-x")
    openai_compat.OpenAICompatProvider(
        base_url="https://api.deepseek.com"
    ).complete("sys", "user")

    assert fake_openai["init"][0]["base_url"] == "https://api.deepseek.com"


def test_openai_compat_pins_temperature_low(monkeypatch, fake_openai):
    from backend.providers import openai_compat

    monkeypatch.setattr(openai_compat.secrets, "get_secret", lambda name: "sk-x")
    openai_compat.OpenAICompatProvider().complete("sys", "user")

    assert fake_openai["calls"][0]["temperature"] == 0.0


def test_openai_compat_never_logs_the_key(monkeypatch, fake_openai, caplog):
    import logging

    from backend.providers import openai_compat

    monkeypatch.setattr(openai_compat.secrets, "get_secret", lambda name: "sk-secret-key")
    with caplog.at_level(logging.DEBUG):
        openai_compat.OpenAICompatProvider().complete("sys", "user")

    assert "sk-secret-key" not in caplog.text


# --- free translation fallback -----------------------------------------------


@pytest.fixture
def fake_deep(monkeypatch):
    state = {"reply": "翻译结果"}

    class FakeTranslator:
        def __init__(self, **kwargs):
            state["init"] = kwargs

        def translate(self, text):
            state["text"] = text
            if isinstance(state["reply"], Exception):
                raise state["reply"]
            return state["reply"]

    module = types.ModuleType("deep_translator")
    module.GoogleTranslator = FakeTranslator
    monkeypatch.setitem(sys.modules, "deep_translator", module)
    return state


def test_free_translate_works(fake_deep):
    from backend.providers.free_translate import FreeTranslateProvider

    assert FreeTranslateProvider().translate("hello") == "翻译结果"


def test_free_translate_is_marked_fallback_only():
    """v1's lesson, design §5.2: the free Google endpoint has no SLA and gets
    rate-limited across a 90-minute lecture, after which v1 silently wrote
    "[translation error]" into the transcript."""
    from backend.providers.free_translate import FreeTranslateProvider

    assert FreeTranslateProvider().fallback_only is True


def test_free_translate_cannot_polish():
    from backend.providers.free_translate import FreeTranslateProvider

    with pytest.raises(llm.LlmError):
        llm.polish(FreeTranslateProvider(), "text")


def test_translate_uses_a_translate_only_provider_directly(fake_deep):
    from backend.providers.free_translate import FreeTranslateProvider

    assert llm.translate(FreeTranslateProvider(), "hello") == "翻译结果"
