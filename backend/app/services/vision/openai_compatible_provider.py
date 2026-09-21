from __future__ import annotations

import base64
import mimetypes
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from .proxy import resolve_proxy
from .base import VisionProvider
from .prompts import CRITIC_PROMPT, SCENE_REPAIR_PROMPT, SCENE_SYSTEM_PROMPT, scene_user_prompt
from .schemas import repair_json
from .test_assets import cleanup_test_image, create_test_image


class VisionProviderError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def image_to_data_url(path: Path) -> str:
    allowed = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = allowed.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    if mime not in allowed.values():
        raise ValueError(f"Unsupported vision image type: {path.suffix or 'unknown'}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class OpenAICompatibleVisionProvider(VisionProvider):
    def __init__(self, *, provider_name: str, api_key: str, base_url: str, model: str, timeout: int = 120, max_retries: int = 1, proxy_url: str | None = None) -> None:
        self.name = provider_name
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.proxy_url = proxy_url if proxy_url is not None else resolve_proxy().url
        if not api_key:
            raise VisionProviderError(f"{provider_name} API Key 未配置")
        if not base_url:
            raise VisionProviderError(f"{provider_name} Base URL 未配置")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VisionProviderError("openai package 未安装") from exc
        http_client = httpx.Client(proxy=self.proxy_url, timeout=timeout, trust_env=False) if self.proxy_url else httpx.Client(timeout=timeout, trust_env=False)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0, http_client=http_client)

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "configured": True, "model": self.model, "baseUrlConfigured": bool(self.base_url)}

    def _request(self, messages: list[dict[str, Any]], *, repair_prompt: str | None = None) -> str:
        request_messages = [*messages, {"role": "user", "content": repair_prompt}] if repair_prompt else messages
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(model=self.model, messages=request_messages)
                return str(response.choices[0].message.content or "")
            except Exception as exc:
                last_error = exc
                code = _status_code(exc)
                if code in {401, 403}:
                    raise VisionProviderError(f"{self.name} API Key 无效", code) from exc
                if attempt < self.max_retries and (code in {408, 429, 500, 502, 503, 504} or code is None):
                    time.sleep(min(2, 2**attempt))
                    continue
                raise VisionProviderError(_safe_error_message(self.name, exc, code), code) from exc
        raise VisionProviderError(_safe_error_message(self.name, last_error, None))

    def _messages(self, image_path: Path, context: dict | None) -> list[dict[str, Any]]:
        return [{"role": "system", "content": SCENE_SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}, {"type": "text", "text": scene_user_prompt((context or {}).get("ocr_elements", []), context)}]}]

    def analyze_scene(self, image_path: Path, context: dict | None = None, mode: str = "standard") -> dict[str, Any]:
        messages = self._messages(image_path, context)
        raw = self._request(messages)
        try:
            return {"provider": self.name, "model": getattr(self, "model", None), "aiUsed": True, **repair_json(raw, "scene")}
        except Exception:
            repaired = self._request(messages, repair_prompt=SCENE_REPAIR_PROMPT)
            return {"provider": self.name, "model": getattr(self, "model", None), "aiUsed": True, **repair_json(repaired, "scene")}

    def critique_reconstruction(self, original_path: Path, reconstructed_path: Path, scene: dict) -> dict[str, Any]:
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(original_path)}}, {"type": "image_url", "image_url": {"url": image_to_data_url(reconstructed_path)}}, {"type": "text", "text": CRITIC_PROMPT + "\nSCENE_SUMMARY:\n" + str({"elements": len(scene.get("elements", [])), "groups": len(scene.get("groups", []))})}]}]
        raw = self._request(messages)
        try:
            return {"provider": self.name, "model": self.model, **repair_json(raw, "critic")}
        except Exception:
            repaired = self._request(messages, repair_prompt=SCENE_REPAIR_PROMPT)
            return {"provider": self.name, "model": self.model, **repair_json(repaired, "critic")}

    def test_connection(self, image_path: Path | None = None) -> dict[str, Any]:
        owned_path = image_path is None
        test_path = image_path
        try:
            if test_path is None:
                test_path = create_test_image()
            messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(test_path)}}, {"type": "text", "text": "Reply exactly with: VISION_OK"}]}]
            answer = self._request(messages)
            return {"success": "VISION_OK" in answer.upper(), "provider": self.name, "model": self.model, "message": "连接成功" if "VISION_OK" in answer.upper() else "模型未返回 VISION_OK", "proxyUrl": self.proxy_url}
        except RuntimeError as exc:
            if str(exc) == "本地测试图片创建失败":
                raise VisionProviderError(str(exc)) from exc
            raise
        finally:
            if owned_path:
                cleanup_test_image(test_path)


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    return int(value) if isinstance(value, int) else None


def _safe_error_message(provider: str, exc: Exception | None, code: int | None) -> str:
    if code == 429:
        return f"{provider} 额度或频率已达到限制"
    if code == 400:
        return f"{provider} 请求格式错误"
    return f"{provider} 请求失败: {type(exc).__name__}" if exc else f"{provider} 请求失败"


__all__ = ["OpenAICompatibleVisionProvider", "VisionProviderError", "image_to_data_url"]
