from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from app.services.settings.runtime_settings import VisionSettings, load_vision_settings

from .base import VisionProvider
from .prompts import CRITIC_PROMPT, SCENE_SYSTEM_PROMPT, scene_user_prompt
from .schemas import repair_json
from .test_assets import cleanup_test_image, create_test_image
from .proxy import resolve_proxy


class GeminiProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GeminiProvider(VisionProvider):
    name = "gemini"

    def __init__(self, vision_settings: VisionSettings | None = None, model: str | None = None, lite: bool = False) -> None:
        runtime = vision_settings or load_vision_settings()
        config = runtime.providers["gemini"]
        self.api_key = config.api_key
        self.model = model or (config.lite_model if lite else config.flash_model) or ("gemini-2.5-flash-lite" if lite else "gemini-2.5-flash")
        proxy = resolve_proxy(runtime.proxy_mode, runtime.manual_proxy)
        self.proxy_url = proxy.url
        if not self.api_key:
            raise GeminiProviderError("Gemini API Key 未配置")
        try:
            from google import genai
        except ImportError as exc:
            raise GeminiProviderError("google-genai 未安装") from exc
        self.client = genai.Client(api_key=self.api_key, http_options=_http_options(self.proxy_url, runtime.timeout))

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "configured": True, "model": self.model}

    def _generate(self, image_path: Path, prompt: str, client: Any | None = None) -> str:
        from google.genai import types
        suffix = image_path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
        try:
            response = (client or self.client).models.generate_content(model=self.model, contents=[prompt, types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime)])
        except Exception as exc:
            # Google currently returns a model-specific 404 for legacy
            # gemini-2.5-flash access on some new API keys. Retry once with
            # the model explicitly recommended by that response.
            if self.model == "gemini-2.5-flash" and _is_legacy_model_error(exc):
                self.model = "gemini-3.6-flash"
                response = (client or self.client).models.generate_content(model=self.model, contents=[prompt, types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime)])
            else:
                raise
        return str(getattr(response, "text", "") or "")

    def analyze_scene(self, image_path: Path, context: dict | None = None, mode: str = "standard") -> dict[str, Any]:
        prompt = SCENE_SYSTEM_PROMPT + "\n" + scene_user_prompt((context or {}).get("ocr_elements", []), context) + "\n只返回JSON。"
        return {"provider": self.name, "model": self.model, "aiUsed": True, **repair_json(self._generate(image_path, prompt), "scene")}

    def critique_reconstruction(self, original_path: Path, reconstructed_path: Path, scene: dict) -> dict[str, Any]:
        from google.genai import types
        parts = [types.Part.from_bytes(data=original_path.read_bytes(), mime_type="image/png"), types.Part.from_bytes(data=reconstructed_path.read_bytes(), mime_type="image/png"), CRITIC_PROMPT + "\n只返回JSON。"]
        response = self.client.models.generate_content(model=self.model, contents=parts)
        return {"provider": self.name, "model": self.model, **repair_json(str(getattr(response, "text", "") or ""), "critic")}

    def test_connection(self, image_path: Path | None = None) -> dict[str, Any]:
        owned_path = image_path is None
        test_path = image_path
        try:
            if test_path is None:
                test_path = create_test_image()
            from google import genai
            from google.genai import types
            # The probe uses a closed, proxy-aware HTTP client and a 60 second
            # timeout. No file handle is retained while the SDK reads the PNG.
            test_client = genai.Client(api_key=self.api_key, http_options=_http_options(self.proxy_url, 60))
            answer = self._generate(test_path, "Reply exactly with: VISION_OK", test_client)
            return {"success": "VISION_OK" in answer.upper(), "provider": self.name, "model": self.model, "message": "连接成功" if "VISION_OK" in answer.upper() else "模型未返回 VISION_OK", "proxyUrl": self.proxy_url}
        except Exception as exc:
            if str(exc) == "本地测试图片创建失败":
                raise GeminiProviderError(str(exc)) from exc
            code = getattr(exc, "status_code", getattr(exc, "code", None))
            if code in {401, 403}:
                message = "Gemini API Key 无效"
            elif code == 429:
                message = "Gemini 免费额度或请求频率已达到限制"
            elif code == 404 and _is_legacy_model_error(exc):
                message = "当前 Gemini 模型不可用，已尝试切换到 gemini-3.6-flash"
            elif any(term in str(exc).lower() for term in ("timeout", "timed out", "deadline", "504")):
                message = "无法连接 Gemini API"
            else:
                message = "Gemini API 请求失败"
            raise GeminiProviderError(message, code if isinstance(code, int) else None) from exc
        finally:
            if owned_path:
                cleanup_test_image(test_path)


__all__ = ["GeminiProvider", "GeminiProviderError"]


def _is_legacy_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "gemini-2.5-flash" in text and ("no longer available" in text or "update your code" in text)


def _http_options(proxy_url: str | None, timeout_seconds: int):
    from google.genai import types

    client_kwargs = {"timeout": timeout_seconds, "trust_env": False}
    if proxy_url:
        client_kwargs["proxy"] = proxy_url
    http_client = httpx.Client(**client_kwargs)
    return types.HttpOptions(
        timeout=timeout_seconds * 1000,
        httpx_client=http_client,
        retry_options=types.HttpRetryOptions(attempts=1),
    )
