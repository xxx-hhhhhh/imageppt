from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.services.reconstruction import pipeline as pipeline_module
from app.services.reconstruction.pipeline import ReconstructionPipeline
from app.services.reconstruction.router import ReconstructionRouter


class FakeStore:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.saved: list[dict] = []

    def get(self, project_id: str) -> dict:
        return {"id": project_id, "images": [{"path": str(self.image_path)}]}

    def save_slide(self, project_id: str, page: int, layout: dict) -> None:
        self.saved.append(layout)


class FakeOCR:
    provider_name = "rapidocr"
    warnings: list[str] = []

    def recognize(self, image_path: Path) -> list:
        return []


class FakeInpainting:
    warnings: list[str] = []
    last_strategies: list[dict] = []
    last_stats = {"ghostingRegionsDetected": 0, "ghostingRegionsRecleaned": 0}

    def restore_background(self, image_path: Path, regions: list, output_path: Path, preserve_regions=None) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_path)

    def reclean_background(self, background_path: Path, bboxes: list) -> int:
        return len(bboxes)


class FakeLayoutService:
    def build_layout(self, image_path: Path, width: int, height: int, regions: list, background_url: str, asset_dir: Path) -> tuple[dict, list]:
        return {
            "version": "1.1",
            "slide": {"width": width, "height": height},
            "elements": [{
                "id": "text_001",
                "type": "text",
                "x": 10,
                "y": 12,
                "width": 160,
                "height": 30,
                "zIndex": 1,
                "text": "OCR original text",
                "confidence": 0.95,
                "style": {"fontSize": 24, "fontWeight": 400},
                "metadata": {},
            }],
        }, []


class FakeVisionProvider:
    warnings: list[str] = []

    def __init__(self, ai_enabled: bool) -> None:
        self.ai_enabled = ai_enabled

    def vision_debug(self) -> dict:
        return {
            "provider": "qwen" if self.ai_enabled else "local",
            "model": "qwen3-vl-flash" if self.ai_enabled else None,
            "rawResponseAvailable": self.ai_enabled,
            "repairUsed": False,
            "normalizationApplied": self.ai_enabled,
            "validationErrors": [],
            "droppedElements": 0,
            "normalizationWarnings": [
                "elements[0].fontWeight: 'bold' -> 700",
                "api_key=sk-secret-value-123456",
            ] if self.ai_enabled else [],
        }

    def critique_reconstruction(self, original_path: Path, reconstructed_path: Path, scene: dict) -> dict:
        return {"issues": [{"elementId": "text_001", "adjustment": {"moveX": 2, "fontSizeScale": 1.05}}]}


class FakeSceneAnalyzer:
    def __init__(self, ai_enabled: bool) -> None:
        self.ai_enabled = ai_enabled
        self.layout_provider = SimpleNamespace(name="fake-layout")
        self.layout_warnings: list[str] = []
        self.vlm_warnings: list[str] = []
        self.vision_provider = FakeVisionProvider(ai_enabled)
        self.vision_routing: dict = {}

    def analyze(self, image_path: Path, layout: dict, regions: list, segmentation: list, enable_vision: bool = True, mode: str = "standard") -> tuple[dict, list[str]]:
        ai_used = self.ai_enabled and enable_vision
        self.vision_routing = {
            "requestedProvider": "qwen" if self.ai_enabled else "local",
            "usedProvider": "qwen" if ai_used else "local",
            "usedModel": "qwen3-vl-flash" if ai_used else None,
            "aiUsed": ai_used,
            "fallbackCount": 0,
        }
        metadata = {
            "visionSemanticType": "text" if ai_used else "unknown",
            "visionMatched": ai_used,
        }
        if ai_used:
            metadata.update({"reconstructionStrategy": "editable_text", "reconstructionStrategySource": "vision"})
        return {
            "version": "2.0",
            "canvas": {"width": layout["slide"]["width"], "height": layout["slide"]["height"]},
            "elements": [{
                "id": "text_001",
                "type": "text",
                "role": "main_title" if ai_used else "body_text",
                "componentType": "text",
                "groupId": "hero" if ai_used else None,
                "bbox": {"left": 10, "top": 12, "width": 160, "height": 30},
                "zIndex": 1,
                "visionConfidence": 0.92 if ai_used else 0.0,
                "finalConfidence": 0.94,
                "text": "AI must not replace OCR text",
                "style": {"fontSize": 24, "fontClass": "display" if ai_used else "sans", "fontWeight": 700 if ai_used else 400},
                "metadata": metadata,
            }],
            "groups": [],
        }, []

    def refine(self, scene: dict) -> dict:
        scene["refined"] = True
        return scene


class FakeSegmentationProvider:
    name = "fake-segmentation"

    def segment(self, image_path: Path, asset_dir: Path, project_id: str) -> list:
        return []


class FakePPTXRenderer:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def render_project(self, project_id: str, layouts: list[dict]) -> tuple[Path, dict]:
        output = self.output_root / project_id / "editable.pptx"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake-pptx")
        return output, {"valid": True}


def _run_pipeline(monkeypatch, tmp_path: Path, ai_enabled: bool) -> tuple[list[dict], dict, dict]:
    image_path = tmp_path / ("ai.png" if ai_enabled else "local.png")
    fixture_path = Path(__file__).parent / "assets" / "component_component_0001.png"
    shutil.copy2(fixture_path, image_path)
    output_root = tmp_path / "outputs"

    def fake_preprocess(source: Path, destination: Path) -> tuple[int, int]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        with Image.open(source) as image:
            return image.size

    def fake_render_preview(background: Path, layout: dict, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(background, destination)
        return destination

    monkeypatch.setattr(pipeline_module, "OUTPUTS_DIR", output_root)
    monkeypatch.setattr(pipeline_module, "OCRService", lambda provider: FakeOCR())
    monkeypatch.setattr(pipeline_module, "InpaintingService", lambda provider: FakeInpainting())
    monkeypatch.setattr(pipeline_module, "preprocess_image", fake_preprocess)
    monkeypatch.setattr(pipeline_module, "render_preview", fake_render_preview)
    monkeypatch.setattr(pipeline_module, "run_visual_qa", lambda *args, **kwargs: {"overall": 1.0})
    monkeypatch.setattr(pipeline_module, "PPTXRenderer", lambda: FakePPTXRenderer(output_root))

    pipeline = object.__new__(ReconstructionPipeline)
    pipeline.store = FakeStore(image_path)
    pipeline.layout_service = FakeLayoutService()
    pipeline.scene_analyzer = FakeSceneAnalyzer(ai_enabled)
    pipeline.segmentation_provider = FakeSegmentationProvider()
    pipeline.segmentation_warnings = []
    pipeline.reconstruction_router = ReconstructionRouter()
    slides, _, _ = pipeline.analyze_project("ai-standard" if ai_enabled else "local-standard", "standard")
    report_path = output_root / ("ai-standard" if ai_enabled else "local-standard") / "conversion_report.json"
    debug_path = output_root / ("ai-standard" if ai_enabled else "local-standard") / "vision_debug.json"
    return slides, json.loads(report_path.read_text(encoding="utf-8")), json.loads(debug_path.read_text(encoding="utf-8"))


def test_standard_mode_applies_qwen_strategy_and_one_critic_round(monkeypatch, tmp_path: Path) -> None:
    slides, report, debug = _run_pipeline(monkeypatch, tmp_path, True)
    element = slides[0]["elements"][0]
    assert element["text"] == "OCR original text"
    assert element["role"] == "main_title"
    assert element["groupId"] == "hero"
    assert element["visionConfidence"] == 0.92
    assert element["metadata"]["reconstructionStrategy"] == "editable_text"
    assert element["style"]["fontClass"] == "serif"
    assert element["style"]["fontRole"] == "main_title"
    assert element["style"]["fontWeight"] == 700
    assert 0 <= element["x"] <= slides[0]["slide"]["width"] - element["width"]
    assert element["style"]["refinedFontSize"] == element["style"]["fontSize"]
    assert abs(element["style"]["fontSizeAdjustment"]) <= element["style"]["estimatedFontSize"] * 0.15
    assert report["aiUsed"] is True
    assert report["visionMatchedElements"] == 1
    assert report["aiStrategiesApplied"] == 1
    assert report["criticRounds"] == 1
    assert report["criticAdjustmentsApplied"] == 1
    assert report["typographyRefined"] is True
    assert report["fontRoleAssignments"] == 1
    assert debug == {
        "provider": "qwen",
        "model": "qwen3-vl-flash",
        "rawResponseAvailable": True,
        "repairUsed": False,
        "normalizationApplied": True,
        "validationErrors": [],
        "droppedElements": 0,
        "normalizationWarnings": ["elements[0].fontWeight: 'bold' -> 700", "api_key=[redacted]"],
    }
    assert "sk-secret-value-123456" not in json.dumps(debug).lower()
    assert "authorization" not in json.dumps(debug).lower()


def test_local_mode_reports_no_ai_strategy_or_critic_round(monkeypatch, tmp_path: Path) -> None:
    _, report, debug = _run_pipeline(monkeypatch, tmp_path, False)
    assert report["aiUsed"] is False
    assert report["visionProvider"] == "local"
    assert report["visionMatchedElements"] == 0
    assert report["aiStrategiesApplied"] == 0
    assert report["criticRounds"] == 0
    assert report["criticAdjustmentsApplied"] == 0
    assert debug["provider"] == "local"
    assert debug["rawResponseAvailable"] is False
