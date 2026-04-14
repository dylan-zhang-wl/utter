import asyncio
from abc import ABC, abstractmethod

from deep_translator import GoogleTranslator as _DeepGoogleTranslator
import openai


# deep-translator uses "zh-CN" rather than "zh-cn"
_LANG_MAP = {"zh-cn": "zh-CN", "zh": "zh-CN"}


def _normalize_lang(lang: str) -> str:
    return _LANG_MAP.get(lang, lang)


class BaseTranslator(ABC):
    @abstractmethod
    async def translate(self, text: str, target_lang: str = "zh-cn") -> str:
        pass


class GoogleTranslator(BaseTranslator):
    async def translate(self, text: str, target_lang: str = "zh-cn") -> str:
        if not text.strip():
            return ""
        target = _normalize_lang(target_lang)
        return await asyncio.to_thread(
            _DeepGoogleTranslator(source="auto", target=target).translate, text
        )


class OpenAITranslator(BaseTranslator):
    def __init__(self, api_key: str):
        self._client = openai.OpenAI(api_key=api_key)

    async def translate(self, text: str, target_lang: str = "zh-cn") -> str:
        if not text.strip():
            return ""
        response = await asyncio.to_thread(
            self._client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate the following English text to Chinese. Return only the translation, nothing else."},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content


def get_translator(engine: str, api_key: str | None = None) -> BaseTranslator:
    if engine == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAITranslator(api_key=api_key)
    return GoogleTranslator()
