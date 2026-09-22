from __future__ import annotations

from pathlib import Path

from PIL import Image

import app.services.ocr.service as ocr_service_module
import app.services.paddle_runtime as paddle_runtime
from app.services.ocr.provider import OCRResult, PaddleOCRProvider
from app.services.ocr.service import OCRService
from app.services.scene.layout_analyzer import AutoLayoutProvider, _normalize_pp_structure
from app.services.scene.scene_analyzer import SceneAnalyzer


class _Result:
    def __init__(self, payload: dict) -> None:
        self.json = {"res": payload}


def test_pp_structure_result_is_normalized_to_project_schema() -> None:
    raw = [_Result({
        "layout_det_res": {
            "boxes": [
                {"label": "paragraph_title", "score": 0.92, "coordinate": [10, 20, 210, 80]},
                {"label": "image", "score": 0.87, "coordinate": [30, 100, 330, 300]},
            ]
        },
        "parsing_res_list": [
            {"block_label": "paragraph_title", "block_bbox": [10, 20, 210, 80], "block_order": 1},
        ],
    })]

    regions = _normalize_pp_structure(raw)

    assert regions == [
        {
            "id": "layout_001",
            "type": "title",
            "role": "title",
            "bbox": {"left": 10.0, "top": 20.0, "width": 200.0, "height": 60.0},
            "confidence": 0.92,
            "source": "pp-structure-v3",
            "readingOrder": 1,
        },
        {
            "id": "layout_002",
            "type": "image",
            "role": "image",
            "bbox": {"left": 30.0, "top": 100.0, "width": 300.0, "height": 200.0},
            "confidence": 0.87,
            "source": "pp-structure-v3",
            "readingOrder": None,
        },
    ]


def test_ocr_service_falls_back_to_rapidocr_on_paddle_inference_failure(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    paddle = PaddleOCRProvider.__new__(PaddleOCRProvider)
    monkeypatch.setattr(paddle, "recognize", lambda _: (_ for _ in ()).throw(RuntimeError("paddle failed")))

    class FakeRapidOCR:
        name = "rapidocr"

        def recognize(self, _: Path) -> list[OCRResult]:
            return [OCRResult("fallback", [0, 0, 10, 10], 0.9, {})]

    monkeypatch.setattr(ocr_service_module, "RapidOCRProvider", FakeRapidOCR)
    service = OCRService.__new__(OCRService)
    service.preferred = "auto"
    service.provider = paddle
    service.warnings = []

    result = service.recognize(image_path)

    assert result[0].text == "fallback"
    assert service.provider_name == "rapidocr"
    assert any("using RapidOCR fallback" in warning for warning in service.warnings)


def test_layout_provider_falls_back_to_opencv_on_pp_structure_failure(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 120), "white").save(image_path)
    provider = AutoLayoutProvider.__new__(AutoLayoutProvider)

    class BrokenStructure:
        name = "pp-structure-v3"

        def analyze(self, _: Path):
            raise RuntimeError("initialization failed")

    from app.services.scene.layout_analyzer import OpenCVLayoutProvider

    provider.primary = BrokenStructure()
    provider.fallback = OpenCVLayoutProvider()
    provider.active = provider.primary
    provider.warnings = []

    regions = provider.analyze(image_path)

    assert isinstance(regions, list)
    assert provider.name == "opencv"
    assert "using OpenCV layout fallback" in provider.warnings[0]


def test_scene_analyzer_passes_layout_regions_to_qwen_context(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (200, 120), "white").save(image_path)
    captured: dict = {}

    class LayoutProvider:
        name = "pp-structure-v3"
        warnings: list[str] = []

        def analyze(self, _: Path):
            return [{"id": "layout_001", "type": "title", "bbox": {"left": 1, "top": 2, "width": 3, "height": 4}, "confidence": 0.9, "source": "pp-structure-v3"}]

    class QwenProvider:
        provider_name = "qwen"
        name = "qwen"
        warnings: list[str] = []

        def analyze_scene(self, _image_path, context=None, mode="standard"):
            captured.update(context or {})
            return {"provider": "qwen", "model": "qwen3-vl-flash", "page": {}, "elements": [], "groups": [], "relations": [], "repeatedComponents": [], "layers": [], "confidence": 0.8, "aiUsed": True}

    analyzer = SceneAnalyzer.__new__(SceneAnalyzer)
    analyzer.layout_provider = LayoutProvider()
    analyzer.layout_warnings = []
    analyzer.vision_provider = QwenProvider()
    analyzer.vision_warnings = []
    analyzer.vlm_provider = analyzer.vision_provider
    analyzer.vlm_warnings = analyzer.vision_warnings
    analyzer.ai_used = False
    analyzer.vision_routing = {}

    analyzer.analyze(image_path, {"slide": {"width": 200, "height": 120}, "elements": []}, [], [])

    assert captured["layout_regions"][0]["source"] == "pp-structure-v3"


def test_paddle_engines_are_process_singletons(monkeypatch) -> None:
    paddle_runtime.reset_paddle_runtime_for_tests()
    created = {"ocr": 0, "structure": 0}

    class FakeOCR:
        def __init__(self, **kwargs) -> None:
            created["ocr"] += 1

    class FakeStructure:
        def __init__(self, **kwargs) -> None:
            created["structure"] += 1

    class FakeModule:
        PaddleOCR = FakeOCR
        PPStructureV3 = FakeStructure

    monkeypatch.setattr(paddle_runtime.importlib, "import_module", lambda _: FakeModule)

    assert paddle_runtime.get_paddle_ocr_engine() is paddle_runtime.get_paddle_ocr_engine()
    assert paddle_runtime.get_pp_structure_v3_engine() is paddle_runtime.get_pp_structure_v3_engine()
    assert created == {"ocr": 1, "structure": 1}
    paddle_runtime.reset_paddle_runtime_for_tests()
