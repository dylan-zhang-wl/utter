"""Ollama — the local LLM, and the reason "fully offline" is a real claim.

Design §2 puts local-first at the centre: unpublished drafts and confidential
translations should not need to leave the machine. Ollama is what makes that the
default rather than an aspiration.
"""

from __future__ import annotations

import logging

from backend.providers.llm import LlmError, httpx

log = logging.getLogger(__name__)

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:4b"


class OllamaProvider:
    id = "ollama"
    display_name = "Ollama (local)"

    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL):
        self.host = host.rstrip("/")
        self.model = model

    def is_available(self) -> tuple[bool, str]:
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            response.raise_for_status()
        except Exception:
            return False, f"Ollama is not reachable at {self.host} — is `ollama serve` running?"
        return True, ""

    def complete(self, system: str, user: str) -> str:
        try:
            response = httpx.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    # Polish must be boring and repeatable. A creative sampler
                    # is exactly how a model starts improving arguments it was
                    # told not to touch (铁律 10).
                    "options": {"temperature": 0.0},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as exc:
            raise LlmError(f"ollama: {exc}") from exc
