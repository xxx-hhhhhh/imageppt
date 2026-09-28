from __future__ import annotations

from pathlib import Path

from app.services.scene.scene_analyzer import SceneAnalyzer
from app.services.segmentation.segmentation_provider import create_segmentation_provider
from app.services.vision.vlm_provider import create_vlm_provider
from app.services.vision.openai_compatible_provider import VisionProviderError
from backend.tests.fixtures.generate_fixtures import generate


def test_fixture_catalog_has_twelve_distinct_families(tmp_path: Path) -> None:
    fixtures = generate(tmp_path / "fixtures")
    assert len(fixtures) == 12
    assert len({path.stem for path in fixtures}) == 12


def test_scene_graph_and_provider_fallback(tmp_path: Path) -> None:
    image = generate(tmp_path / "fixtures")[1]
    analyzer = SceneAnalyzer("auto", "none")
    layout = {"slide": {"width": 960, "height": 540}, "elements": [{"id": "shape", "type": "roundedRectangle", "x": 30, "y": 40, "width": 200, "height": 100, "rotation": 0, "zIndex": 5, "style": {"fill": "#DCE6F1"}}]}
    scene, warnings = analyzer.analyze(image, layout, [], [])
    refined = analyzer.refine(scene)
    assert {"canvas", "regions", "elements", "groups", "relations", "styleTokens", "confidence"}.issubset(refined)
    assert refined["elements"][0]["bbox"]["left"] >= 0
    assert analyzer.layout_provider.name in {"opencv", "pp-structure-v3"}
    assert isinstance(warnings, list)
    segmentation, _ = create_segmentation_provider("auto")
    assert segmentation.name == "opencv"
    vlm, _ = create_vlm_provider("none")
    assert vlm.name == "none"


def test_scene_analyzer_reports_concrete_qwen_validation_fallback(tmp_path: Path) -> None:
    class InvalidQwenProvider:
        provider_name = "qwen"
        name = "qwen"
        warnings: list[str] = []

        def analyze_scene(self, image_path, context=None, mode="standard"):
            raise VisionProviderError("Vision validation failed: elements[3].visualComplexity: invalid value")

    image = generate(tmp_path / "fixtures")[1]
    analyzer = SceneAnalyzer("auto", "none")
    analyzer.vision_provider = InvalidQwenProvider()
    analyzer.vision_warnings = []
    analyzer.vlm_warnings = analyzer.vision_warnings
    layout = {
        "slide": {"width": 960, "height": 540},
        "elements": [{
            "id": "shape",
            "type": "roundedRectangle",
            "x": 30,
            "y": 40,
            "width": 200,
            "height": 100,
            "zIndex": 5,
            "style": {"fill": "#DCE6F1"},
        }],
    }

    scene, warnings = analyzer.analyze(image, layout, [], [])

    assert scene["vision"]["provider"] == "local"
    assert analyzer.vision_routing["usedProvider"] == "local"
    assert analyzer.vision_routing["usedModel"] is None
    assert analyzer.vision_routing["fallbackCount"] == 1
    assert analyzer.vision_routing["aiUsed"] is False
    assert "elements[3].visualComplexity" in analyzer.vision_routing["attempts"][0]["error"]
    assert any("elements[3].visualComplexity" in warning for warning in warnings)


def test_partial_qwen_plan_keeps_uncovered_ocr_text_once(tmp_path: Path) -> None:
    class PartialProvider:
        provider_name = "qwen"
        name = "qwen"
        warnings: list[str] = []

        def analyze_scene(self, image_path, context=None, mode="high"):
            return {"provider": "qwen", "model": "mock", "aiUsed": True, "elements": [],
                    "reconstructionPlan": {"modules": [{"id": "visual", "bbox": {"left": 0, "top": 0, "width": 1, "height": 0.6}, "confidence": 0.9}]},
                    "planCoverage": {"textCoverage": 0.6, "uncoveredTextIds": ["text_001", "text_005"], "status": "partial"}}

    image = generate(tmp_path / "fixtures")[1]
    analyzer = SceneAnalyzer("opencv", "none")
    analyzer.vision_provider = PartialProvider()
    analyzer.vision_warnings = []
    analyzer.vlm_warnings = analyzer.vision_warnings
    elements = [{"id": f"text_{index:03d}", "type": "text", "x": 20, "y": index * 35, "width": 120, "height": 25, "zIndex": 20, "text": f"Line {index}", "style": {"fontSize": 20}, "metadata": {}} for index in range(1, 6)]
    scene, _ = analyzer.analyze(image, {"slide": {"width": 960, "height": 540}, "elements": elements}, [], [], mode="high")

    assert scene["vision"]["aiUsed"] is True
    assert len([item for item in scene["elements"] if item["type"] == "text"]) == 5
    assert len({item["id"] for item in scene["elements"]}) == 5
    assert all(next(item for item in scene["elements"] if item["id"] == item_id)["metadata"]["reconstructionStrategy"] == "editable_text" for item_id in ("text_001", "text_005"))
