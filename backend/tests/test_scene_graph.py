from __future__ import annotations

from pathlib import Path

from app.services.scene.scene_analyzer import SceneAnalyzer
from app.services.segmentation.segmentation_provider import create_segmentation_provider
from app.services.vision.vlm_provider import create_vlm_provider
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
