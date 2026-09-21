from pathlib import Path

from app.services.settings.runtime_settings import ProviderSettings, VisionSettings
from app.services.vision.base import VisionProvider
from app.services.vision.openai_compatible_provider import VisionProviderError
from app.services.vision.router import VisionRouter


class FakeQwenProvider(VisionProvider):
    def __init__(self, name: str, model: str, fail: bool = False):
        self.name, self.model, self.fail = name, model, fail

    def analyze_scene(self, image_path: Path, context=None, mode="standard"):
        if self.fail:
            raise VisionProviderError("quota", 429)
        return {"provider": "qwen", "model": self.model, "aiUsed": True, "elements": []}

    def critique_reconstruction(self, original_path, reconstructed_path, scene):
        return {"issues": []}


def qwen_settings(**kwargs):
    config = ProviderSettings(enabled=True, api_key="qwen-key", base_url="https://qwen.test/v1", model="qwen3-vl-flash")
    config = kwargs.get("qwen", config)
    return VisionSettings(selected_provider="qwen", providers={"qwen": config})


def test_router_calls_only_qwen(tmp_path):
    image = tmp_path / "tiny.png"
    image.write_bytes(b"png")
    called = []
    router = VisionRouter(qwen_settings(), provider_factory=lambda name, model: called.append((name, model)) or FakeQwenProvider(name, model or "qwen3-vl-flash"))
    result = router.analyze_scene(image, {}, "standard")
    assert result["provider"] == "qwen"
    assert result["aiUsed"] is True
    assert called == [("qwen", "qwen3-vl-flash")]
    assert result["routing"]["fallbackCount"] == 0


def test_router_does_not_fallback_to_other_provider(tmp_path):
    image = tmp_path / "tiny.png"
    image.write_bytes(b"png")
    called = []
    router = VisionRouter(qwen_settings(), provider_factory=lambda name, model: called.append(name) or FakeQwenProvider(name, model or "qwen3-vl-flash", fail=True))
    result = router.analyze_scene(image, {}, "standard")
    assert result["provider"] == "local"
    assert result["aiUsed"] is False
    assert result["routing"]["usedModel"] is None
    assert result["routing"]["fallbackCount"] == 1
    assert called == ["qwen"]


def test_router_local_mode_never_calls_qwen(tmp_path):
    image = tmp_path / "tiny.png"
    image.write_bytes(b"png")
    called = []
    settings = qwen_settings()
    settings.enabled = False
    settings.selected_provider = "local"
    router = VisionRouter(settings, provider_factory=lambda name, model: called.append(name) or FakeQwenProvider(name, model or "qwen3-vl-flash"))
    result = router.analyze_scene(image, {}, "standard")
    assert result["provider"] == "local"
    assert called == []
