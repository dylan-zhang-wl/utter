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
DEFAULT_MODEL = "gpt-4o-mini"
SECRET_NAME = "openai_api_key"


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
            return False, f"no API key stored for {self.secret_name} (add one in settings)"
        return True, ""

    def complete(self, system: str, user: str) -> str:
        key = self._key()
        if not key:
            raise LlmError(f"no API key stored for {self.secret_name}")

        try:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=self.base_url)
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
