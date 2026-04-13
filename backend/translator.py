import asyncio
from abc import ABC, abstractmethod

try:
    import googletrans
    _GOOGLETRANS_AVAILABLE = True
except (ImportError, AttributeError):
    googletrans = None
    _GOOGLETRANS_AVAILABLE = False

import openai


class BaseTranslator(ABC):
    @abstractmethod
    async def translate(self, text: str, target_lang: str = "zh-cn") -> str:
        pass


class GoogleTranslator(BaseTranslator):
    def __init__(self):
        if _GOOGLETRANS_AVAILABLE:
            self._translator = googletrans.Translator()
        else:
            self._translator = None

    async def translate(self, text: str, target_lang: str = "zh-cn") -> str:
        if not text.strip():
            return ""
        if not _GOOGLETRANS_AVAILABLE:
            raise RuntimeError(
                "googletrans is not available. Install a compatible version: "
                "pip install googletrans==4.0.0rc1"
            )
        result = await asyncio.to_thread(
            self._translator.translate, text, dest=target_lang
        )
        return result.text


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
