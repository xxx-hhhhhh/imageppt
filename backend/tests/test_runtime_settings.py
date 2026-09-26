from __future__ import annotations

from app.services.settings import runtime_settings


def test_runtime_settings_stay_outside_project_and_mask_key(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(runtime_settings, "_settings_path", lambda: path)
    saved = runtime_settings.save_vision_settings({"api_key": "sk-abcdefgh123456", "base_url": "https://example.test/v1"})
    assert saved.api_key == "sk-abcdefgh123456"
    preserved = runtime_settings.save_vision_settings({"api_key": ""})
    assert preserved.api_key == saved.api_key
    masked = runtime_settings.save_vision_settings({"api_key": runtime_settings.mask_api_key(saved.api_key)})
    assert masked.api_key == saved.api_key
    assert runtime_settings.mask_api_key(saved.api_key) == "sk-ab*********456"
    assert path.exists()


def test_qwen_endpoint_selection_is_explicit_even_with_workspace_key(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_settings, "_settings_path", lambda: tmp_path / "settings.json")
    key = "sk-ws-example-placeholder"
    china = runtime_settings.save_vision_settings({"api_key": key, "base_url": runtime_settings.QWEN_BASE_URL})
    assert china.base_url == runtime_settings.QWEN_BASE_URL
    international = runtime_settings.save_vision_settings({"base_url": runtime_settings.QWEN_CLOUD_BASE_URL})
    assert international.base_url == runtime_settings.QWEN_CLOUD_BASE_URL
    assert runtime_settings.load_vision_settings().base_url == runtime_settings.QWEN_CLOUD_BASE_URL
