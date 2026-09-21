import pytest

from app.services.settings.runtime_settings import ProviderSettings, VisionSettings
from app.services.vision.gemini_provider import GeminiProvider, GeminiProviderError


def test_gemini_without_key_is_safe_fallback():
    settings = VisionSettings(providers={"gemini": ProviderSettings(enabled=True), "qwen": ProviderSettings(), "openrouter": ProviderSettings(), "custom": ProviderSettings()})
    with pytest.raises(GeminiProviderError, match="API Key"):
        GeminiProvider(settings)
