from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"


def ensure_runtime_dirs() -> None:
    for directory in (UPLOADS_DIR, OUTPUTS_DIR, TEMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_UPLOAD_MB = env_int("MAX_UPLOAD_MB", 25)
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "auto")
INPAINT_PROVIDER = os.getenv("INPAINT_PROVIDER", "opencv")
LAYOUT_PROVIDER = os.getenv("LAYOUT_PROVIDER", "auto")
SEGMENTATION_PROVIDER = os.getenv("SEGMENTATION_PROVIDER", "auto")
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "none")
VISION_PROVIDER = os.getenv("VISION_PROVIDER", VLM_PROVIDER if VLM_PROVIDER != "none" else "qwen")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.6-flash")
GEMINI_FLASH_LITE_MODEL = os.getenv("GEMINI_FLASH_LITE_MODEL", "gemini-2.5-flash-lite")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl-flash")
QWEN_ENABLE_THINKING = os.getenv("QWEN_ENABLE_THINKING", "false").lower() == "true"
QWEN_TIMEOUT = env_int("QWEN_TIMEOUT", 120)
QWEN_MAX_RETRIES = env_int("QWEN_MAX_RETRIES", 2)
CONVERSION_MODE = os.getenv("CONVERSION_MODE", "standard")
MAX_AI_CALLS_PER_SLIDE = env_int("MAX_AI_CALLS_PER_SLIDE", 3)
PUBLIC_SHARED_MODE = os.getenv("IMAGE2EDITABLEPPT_PUBLIC_SHARED", "false").strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    QWEN_API_KEY = QWEN_API_KEY
    QWEN_BASE_URL = QWEN_BASE_URL
    QWEN_MODEL = QWEN_MODEL
    QWEN_ENABLE_THINKING = QWEN_ENABLE_THINKING
    QWEN_TIMEOUT = QWEN_TIMEOUT
    QWEN_MAX_RETRIES = QWEN_MAX_RETRIES
    GEMINI_API_KEY = GEMINI_API_KEY
    GEMINI_FLASH_MODEL = GEMINI_FLASH_MODEL
    GEMINI_FLASH_LITE_MODEL = GEMINI_FLASH_LITE_MODEL
    OPENROUTER_API_KEY = OPENROUTER_API_KEY
    OPENROUTER_BASE_URL = OPENROUTER_BASE_URL
    OPENROUTER_MODEL = OPENROUTER_MODEL
    MAX_AI_CALLS_PER_SLIDE = MAX_AI_CALLS_PER_SLIDE


settings = Settings()
