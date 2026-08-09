"""Keyless translation, as a last resort only.

v1 used deep-translator's free Google endpoint as its default and design §5.2
records what happened: no SLA, rate-limited after a few hundred calls, and a
90-minute lecture makes several hundred. Once throttled, v1 silently wrote
"[translation error]" into the transcript — the user found out afterwards, from
the archive, with the lecture over.

So this exists, and it is never the default. `fallback_only` is read by the
settings UI to say so out loud.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class FreeTranslateProvider:
    id = "free_translate"
    display_name = "Free translation (no key, no guarantees)"

    #: Translation only — there is no model here to run a polish prompt through.
    can_polish = False

    #: Never select this automatically. v1's lesson, design §5.2.
    fallback_only = True

    def __init__(self, source: str = "en", target: str = "zh-CN"):
        self.source = source
        self.target = target

    def is_available(self) -> tuple[bool, str]:
        try:
            import deep_translator  # noqa: F401
        except ImportError:
            return False, "deep-translator is not installed"
        return True, ""

    def translate(self, text: str) -> str:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source=self.source, target=self.target).translate(text)
