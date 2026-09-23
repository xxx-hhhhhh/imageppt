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


def test_qwen_cloud_workspace_key_uses_its_own_official_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_settings, "_settings_path", lambda: tmp_path / "settings.json")
    saved = runtime_settings.save_vision_settings({"api_key": "sk-ws-example-placeholder", "base_url": runtime_settings.QWEN_BASE_URL})
    assert saved.base_url == runtime_settings.QWEN_CLOUD_BASE_URL
    assert runtime_settings.load_vision_settings().base_url == runtime_settings.QWEN_CLOUD_BASE_URL
