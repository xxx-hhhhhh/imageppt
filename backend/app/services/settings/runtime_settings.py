from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _settings_path() -> Path:
    if platform.system().lower() == "windows":
        root = Path(os.getenv("LOCALAPPDATA", Path.home()))
        return root / "Image2EditablePPT" / "settings.json"
    return Path.home() / ".image2editableppt" / "settings.json"


@dataclass
class ProviderSettings:
    enabled: bool = False
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    flash_model: str = ""
    lite_model: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None, defaults: "ProviderSettings" | None = None) -> "ProviderSettings":
        current = defaults or cls()
        raw = value or {}
        aliases = {
            "api_key": "apiKey", "base_url": "baseUrl", "flash_model": "flashModel",
            "lite_model": "liteModel",
        }
        for key in ("enabled", "api_key", "base_url", "model", "flash_model", "lite_model"):
            source = key if key in raw else aliases.get(key, key)
            if source in raw and raw[source] is not None:
                setattr(current, key, raw[source])
        return current

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "apiKey": self.api_key,
            "baseUrl": self.base_url,
            "model": self.model,
            "flashModel": self.flash_model,
            "liteModel": self.lite_model,
        }


@dataclass
class VisionSettings:
    enabled: bool = True
    selected_provider: str = "qwen"
    mode: str = "standard"
    timeout: int = 120
    proxy_mode: str = "system"
    manual_proxy: str = ""
    providers: dict[str, ProviderSettings] | None = None

    def __post_init__(self) -> None:
        self.providers = self.providers or default_providers()

    # Backwards-compatible accessors for the previous single-Qwen API.
    @property
    def provider(self) -> str:
        return self.selected_provider

    @provider.setter
    def provider(self, value: str) -> None:
        self.selected_provider = value

    @property
    def api_key(self) -> str:
        return self.providers["qwen"].api_key

    @api_key.setter
    def api_key(self, value: str) -> None:
        self.providers["qwen"].api_key = value

    @property
    def base_url(self) -> str:
        return self.providers["qwen"].base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self.providers["qwen"].base_url = value

    @property
    def model(self) -> str:
        return self.providers["qwen"].model

    @model.setter
    def model(self, value: str) -> None:
        self.providers["qwen"].model = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "selectedProvider": "qwen" if self.selected_provider not in {"local", "none", "off"} else "local",
            "mode": self.mode,
            "timeout": self.timeout,
            "proxyMode": self.proxy_mode,
            "manualProxy": self.manual_proxy,
            "providers": {"qwen": self.providers["qwen"].to_dict()},
        }


def default_providers() -> dict[str, ProviderSettings]:
    from app.config import QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL

    return {"qwen": ProviderSettings(enabled=bool(QWEN_API_KEY), api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL, model=QWEN_MODEL or "qwen3-vl-flash")}


def load_vision_settings() -> VisionSettings:
    from app.config import VISION_PROVIDER, QWEN_TIMEOUT

    value = VisionSettings(enabled=VISION_PROVIDER not in {"none", "off", "false"}, selected_provider="local" if VISION_PROVIDER in {"none", "off", "false", "local"} else "qwen", timeout=QWEN_TIMEOUT)
    path = _settings_path()
    try:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            # The public contract stores settings under ``vision``.  Read the
            # previous flat format as well so existing local settings survive
            # the upgrade without exposing the secret in the API response.
            saved = raw.get("vision", raw) if isinstance(raw, dict) else {}
            value.enabled = saved.get("enabled", value.enabled)
            saved_provider = saved.get("selectedProvider", saved.get("provider", "qwen"))
            value.selected_provider = "local" if saved_provider in {"local", "none", "off", "local_only"} or saved.get("enabled") is False else "qwen"
            value.mode = saved.get("mode", value.mode)
            value.timeout = int(saved.get("timeout", value.timeout))
            value.proxy_mode = saved.get("proxyMode", saved.get("proxy_mode", value.proxy_mode))
            value.manual_proxy = saved.get("manualProxy", saved.get("manual_proxy", value.manual_proxy))
            saved_providers = saved.get("providers", {}) or {}
            if isinstance(saved_providers, dict) and isinstance(saved_providers.get("qwen"), dict):
                value.providers["qwen"] = ProviderSettings.from_dict(saved_providers["qwen"], value.providers["qwen"])
            # Migrate the previous single-provider flat settings format.
            legacy = saved
            if any(key in legacy for key in ("api_key", "apiKey", "base_url", "baseUrl", "model")):
                value.providers["qwen"] = ProviderSettings.from_dict(legacy, value.providers["qwen"])
                value.providers["qwen"].enabled = value.providers["qwen"].enabled and value.selected_provider == "qwen"
    except (OSError, ValueError, TypeError):
        pass
    if not value.providers["qwen"].base_url:
        value.providers["qwen"].base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not value.providers["qwen"].model:
        value.providers["qwen"].model = "qwen3-vl-flash"
    value.providers["qwen"].enabled = value.selected_provider == "qwen"
    return value


def save_vision_settings(payload: VisionSettings | dict[str, Any]) -> VisionSettings:
    current = load_vision_settings()
    incoming = payload.to_dict() if isinstance(payload, VisionSettings) else dict(payload)
    current.enabled = incoming.get("enabled", current.enabled)
    incoming_provider = incoming.get("selectedProvider", incoming.get("provider", current.selected_provider))
    current.selected_provider = "local" if incoming_provider in {"local", "none", "off", "local_only"} or incoming.get("enabled") is False else "qwen"
    current.mode = incoming.get("mode", current.mode)
    current.timeout = int(incoming.get("timeout", current.timeout))
    current.proxy_mode = incoming.get("proxyMode", incoming.get("proxy_mode", current.proxy_mode))
    current.manual_proxy = incoming.get("manualProxy", incoming.get("manual_proxy", current.manual_proxy))
    providers = incoming.get("providers") or {}
    config = providers.get("qwen") if isinstance(providers, dict) else None
    if isinstance(config, dict):
        raw = dict(config)
        if raw.pop("clearApiKey", False) or raw.pop("clear_api_key", False):
            current.providers["qwen"].api_key = ""
        # Empty API keys preserve an existing secret; masked values are never
        # accepted as replacements.
        if "apiKey" in raw and (not raw["apiKey"] or str(raw["apiKey"]).startswith(("sk-****", "AIza****"))):
            raw.pop("apiKey")
        if "api_key" in raw and not raw["api_key"]:
            raw.pop("api_key")
        current.providers["qwen"] = ProviderSettings.from_dict(raw, current.providers["qwen"])
    # Accept the old flat payload while migrating callers.
    if any(key in incoming for key in ("api_key", "apiKey", "base_url", "baseUrl")):
        raw = dict(incoming)
        if not raw.get("api_key", raw.get("apiKey", "")):
            raw.pop("api_key", None); raw.pop("apiKey", None)
        current.providers["qwen"] = ProviderSettings.from_dict(raw, current.providers["qwen"])
    current.providers["qwen"].enabled = current.selected_provider == "qwen"
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vision": current.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return current


def mask_api_key(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return value[:2] + "*" * max(1, len(value) - 3) + value[-1:]
    return f"{value[:5]}{'*' * max(4, len(value) - 8)}{value[-3:]}"


def settings_path() -> Path:
    """Exposed for tests without making the path part of the API response."""
    return _settings_path()
