import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from backend.translator import GoogleTranslator, OpenAITranslator, get_translator


def test_get_translator_google():
    t = get_translator("google")
    assert isinstance(t, GoogleTranslator)


def test_get_translator_openai():
    t = get_translator("openai", api_key="test-key")
    assert isinstance(t, OpenAITranslator)


@pytest.mark.asyncio
async def test_google_translate():
    mock_googletrans = MagicMock()
    mock_instance = MagicMock()
    mock_instance.translate.return_value = MagicMock(text="你好世界")
    mock_googletrans.Translator.return_value = mock_instance

    with patch("backend.translator.googletrans", mock_googletrans), \
         patch("backend.translator._GOOGLETRANS_AVAILABLE", True):
        t = GoogleTranslator()
        result = await t.translate("Hello world")
        assert result == "你好世界"


@pytest.mark.asyncio
async def test_openai_translate():
    with patch("backend.translator.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="你好世界"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_cls.return_value = mock_client

        t = OpenAITranslator(api_key="test-key")
        result = await t.translate("Hello world")
        assert result == "你好世界"


@pytest.mark.asyncio
async def test_translate_empty_string():
    t = GoogleTranslator()
    result = await t.translate("")
    assert result == ""
