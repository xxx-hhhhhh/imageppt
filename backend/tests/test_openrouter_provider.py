import pytest

from app.services.settings.runtime_settings import ProviderSettings, VisionSettings
from app.services.vision.openrouter_provider import OpenRouterProvider
from app.services.vision.openai_compatible_provider import VisionProviderError


def test_openrouter_without_key_does_not_make_network_call():
    settings = VisionSettings(providers={"gemini": ProviderSettings(), "qwen": ProviderSettings(), "openrouter": ProviderSettings(base_url="https://openrouter.ai/api/v1"), "custom": ProviderSettings()})
    with pytest.raises(VisionProviderError, match="API Key"):
        OpenRouterProvider(settings)
