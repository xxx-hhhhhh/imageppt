from __future__ import annotations

import importlib
import importlib.util
import threading
from importlib import metadata
from typing import Any


_state_lock = threading.RLock()
_ocr_predict_lock = threading.Lock()
_structure_predict_lock = threading.Lock()
_ocr_engine: Any | None = None
_structure_engine: Any | None = None
_ocr_error: str | None = None
_structure_error: str | None = None
_active_layout_provider = "opencv"


def paddleocr_installed() -> bool:
    return importlib.util.find_spec("paddleocr") is not None


def pp_structure_v3_available() -> bool:
    if not paddleocr_installed():
        return False
    try:
        version = metadata.version("paddleocr")
        return int(version.split(".", 1)[0]) >= 3
    except (metadata.PackageNotFoundError, TypeError, ValueError):
        return False


def get_paddle_ocr_engine() -> Any:
    global _ocr_engine, _ocr_error
    if _ocr_engine is not None:
        return _ocr_engine
    with _state_lock:
        if _ocr_engine is not None:
            return _ocr_engine
        try:
            module = importlib.import_module("paddleocr")
            engine_type = getattr(module, "PaddleOCR")
            _ocr_engine = engine_type(
                lang="ch",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            _ocr_error = None
            return _ocr_engine
        except Exception as exc:
            _ocr_error = f"{type(exc).__name__}: {exc}"
            raise


def get_pp_structure_v3_engine() -> Any:
    global _structure_engine, _structure_error
    if _structure_engine is not None:
        return _structure_engine
    with _state_lock:
        if _structure_engine is not None:
            return _structure_engine
        try:
            module = importlib.import_module("paddleocr")
            engine_type = getattr(module, "PPStructureV3")
            _structure_engine = engine_type(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                use_seal_recognition=False,
                use_table_recognition=False,
                use_formula_recognition=False,
                use_chart_recognition=False,
                use_region_detection=False,
            )
            _structure_error = None
            return _structure_engine
        except Exception as exc:
            _structure_error = f"{type(exc).__name__}: {exc}"
            raise


def predict_ocr(image_path: str) -> list[Any]:
    engine = get_paddle_ocr_engine()
    with _ocr_predict_lock:
        return list(engine.predict(image_path))


def predict_structure(image_path: str) -> list[Any]:
    global _active_layout_provider
    engine = get_pp_structure_v3_engine()
    with _structure_predict_lock:
        result = list(engine.predict(image_path))
    _active_layout_provider = "pp-structure-v3"
    return result


def set_active_layout_provider(name: str) -> None:
    global _active_layout_provider
    _active_layout_provider = name


def paddle_diagnostics(layout_provider: str | None = None) -> dict[str, Any]:
    return {
        "paddleocrInstalled": paddleocr_installed(),
        "ppStructureV3Available": pp_structure_v3_available(),
        "ppStructureV3Initialized": _structure_engine is not None,
        "layoutProvider": layout_provider or _active_layout_provider,
        "paddleOCREngineInitialized": _ocr_engine is not None,
        "paddleOCRError": _ocr_error,
        "ppStructureV3Error": _structure_error,
    }


def reset_paddle_runtime_for_tests() -> None:
    global _ocr_engine, _structure_engine, _ocr_error, _structure_error, _active_layout_provider
    with _state_lock:
        _ocr_engine = None
        _structure_engine = None
        _ocr_error = None
        _structure_error = None
        _active_layout_provider = "opencv"


__all__ = [
    "get_paddle_ocr_engine",
    "get_pp_structure_v3_engine",
    "paddle_diagnostics",
    "paddleocr_installed",
    "pp_structure_v3_available",
    "predict_ocr",
    "predict_structure",
    "reset_paddle_runtime_for_tests",
    "set_active_layout_provider",
]
