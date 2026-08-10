"""One provider for every OpenAI-shaped endpoint.

OpenAI, DeepSeek, Groq, 硅基流动 and most others speak the same API, so a single
implementation plus a base_url covers all of them. BYOK throughout: the key
comes from the Keychain, never from a config file, and never ships in the build.
"""

from __future__ import annotations

import logging

from backend import secrets
from backend.providers.llm import LlmError

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
SECRET_NAME = "openai_api_key"

#: Tried in order; the first one the key can actually reach wins.
#:
#: Hardcoding a single id is how a working build dies six months later with
#: `404 model not found` — gpt-4o-mini was this file's default until the day
#: OpenAI's lineup had moved on twice. Names below are candidates, not
#: promises: `resolve_model` asks the key what exists, and `utter polish
#: --benchmark` times whatever it finds. Cheapest-and-fastest first, because
#: polish is punctuation and nothing else.
PREFERRED_MODELS = (
    "gpt-5.4-nano",
    "gpt-5-nano",
    "gpt-5.4-mini",
    "gpt-5-mini",
    "gpt-4.1-nano",
    "gpt-4o-mini",
)
DEFAULT_MODEL = PREFERRED_MODELS[-1]

#: Anything matching these is not a chat model and must not be benchmarked or
#: chosen: embeddings, speech, images, moderation.
_NOT_CHAT = ("embedding", "whisper", "tts", "dall-e", "moderation", "audio", "image",
             "realtime", "transcribe", "search", "codex", "sora")


class OpenAICompatProvider:
    id = "openai_compat"
    display_name = "OpenAI-compatible API (bring your own key)"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        secret_name: str = SECRET_NAME,
    ):
        self.base_url = base_url
        self.model = model
        self.secret_name = secret_name
        #: An explicit model is honoured as given; a default gets resolved
        #: against the key on first use.
        self._resolved = model != DEFAULT_MODEL

    def _key(self) -> str | None:
        # 铁律 4. The config model does not even carry this field any more.
        return secrets.get_secret(self.secret_name)

    def is_available(self) -> tuple[bool, str]:
        """Key presence only — deliberately no test request.

        `utter doctor` probes every provider, and a probe that called the API
        would cost the user money and a round trip every time they asked what
        was wrong.
        """
        if not self._key():
            # Name the command. "add one in settings" pointed at a settings
            # screen that does not exist.
            return False, f"钥匙串里没有 {self.secret_name} —— 跑 `utter key {self.secret_name}` 存一个"
        return True, ""

    def _client(self):
        key = self._key()
        if not key:
            raise LlmError(f"no API key stored for {self.secret_name}")
        from openai import OpenAI

        return OpenAI(api_key=key, base_url=self.base_url)

    def available_models(self) -> list[str]:
        """What this key can actually reach. One network call."""
        try:
            return [m.id for m in self._client().models.list()]
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(f"{self.base_url}: {type(exc).__name__}") from None

    def chat_models(self) -> list[str]:
        """Reachable models that could plausibly answer a prompt.

        The list a key can see includes embeddings, speech and image models,
        and asking one of those to punctuate a sentence produces a confusing
        error rather than an obviously wrong answer.
        """
        return sorted(
            m for m in self.available_models()
            if not any(word in m.lower() for word in _NOT_CHAT)
        )

    def resolve_model(self) -> str:
        """Pick the best reachable model, once per process.

        A failed listing keeps the compiled-in default and lets the real
        request produce the real error — 铁律 8, a polish problem costs the
        polish, never the words.
        """
        if self._resolved:
            return self.model
        try:
            reachable = set(self.available_models())
            for candidate in PREFERRED_MODELS:
                if candidate in reachable:
                    self.model = candidate
                    break
            else:
                log.warning("none of the preferred models exist; using %s", self.model)
        except LlmError as exc:
            log.warning("could not list models (%s), using %s", exc, self.model)
        self._resolved = True
        log.info("model: %s", self.model)
        return self.model

    def complete(self, system: str, user: str) -> str:
        key = self._key()
        if not key:
            raise LlmError(f"no API key stored for {self.secret_name}")

        self.resolve_model()
        try:
            client = self._client()
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.0,  # see 铁律 10 — no creativity wanted here
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            # No exc_info and no repr of the client: an SDK exception can carry
            # the request headers, and those hold the key.
            raise LlmError(f"{self.base_url}: {type(exc).__name__}") from None
