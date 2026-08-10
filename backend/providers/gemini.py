"""Gemini, through Google's OpenAI-compatible endpoint.

Not a new client. Google publishes an OpenAI-shaped API at
`/v1beta/openai/`, so this is `OpenAICompatProvider` with a base URL, a
different Keychain entry, and a model chosen for the job.

## Why Flash-Lite by default

Polish is not a hard task. 铁律 10 allows exactly three operations —
punctuation, filler removal, paragraphing — and explicitly forbids the
model from thinking about the argument. A reasoning model here would be
paying for capability that the prompt spends its whole length suppressing.

Flash-Lite is also the only tier whose free quota fits real dictation. As of
2026 the free limits are roughly 15 requests/minute and 1000/day for
Flash-Lite against 10/minute and 250/day for Flash — and 铁律 11 already
serialises our calls, so requests per minute is the binding constraint, not
tokens.

## The model id is checked, not assumed

Google retires model ids on a schedule, and a provider that fails with
`404 model not found` a month after it was written is worse than one that
says which models the key can actually reach. `available_models()` asks.
"""

from __future__ import annotations

import logging

from backend.providers.llm import LlmError
from backend.providers.openai_compat import OpenAICompatProvider

log = logging.getLogger(__name__)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
SECRET_NAME = "google_api_key"

#: Tried in order; the first one the key can actually reach wins. Newest first,
#: because Google keeps the previous generation alive well past the release of
#: the next, and a build from six months ago should still start.
PREFERRED_MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
)


class GeminiProvider(OpenAICompatProvider):
    id = "gemini"
    display_name = "Gemini (Google, 免费档够用)"

    def __init__(self, model: str | None = None):
        super().__init__(
            base_url=BASE_URL,
            model=model or PREFERRED_MODELS[-2],
            secret_name=SECRET_NAME,
        )
        self._explicit = model is not None
        self._resolved = model is not None

    def available_models(self) -> list[str]:
        """What this key can actually reach. One network call."""
        key = self._key()
        if not key:
            raise LlmError(f"no API key stored for {self.secret_name}")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=self.base_url)
            return [m.id.removeprefix("models/") for m in client.models.list()]
        except Exception as exc:
            raise LlmError(f"{self.base_url}: {type(exc).__name__}") from None

    def resolve_model(self) -> str:
        """Pick the best reachable model, once per process.

        A failure here is not fatal — it falls back to the compiled-in default
        and lets the actual request produce the real error. Refusing to polish
        because a model *listing* failed would be the wrong trade: 铁律 8 says
        a polish problem costs the polish, never the words.
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
                log.warning(
                    "none of %s是可用的，沿用 %s", PREFERRED_MODELS, self.model
                )
        except LlmError as exc:
            log.warning("could not list Gemini models (%s), using %s", exc, self.model)
        self._resolved = True
        log.info("Gemini model: %s", self.model)
        return self.model

    def complete(self, system: str, user: str) -> str:
        self.resolve_model()
        return super().complete(system, user)
