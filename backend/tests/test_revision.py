from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.models.project_store import ProjectStore
from app import main as api_main
from fastapi.testclient import TestClient
from app.services.reconstruction import revision
from app.services.reconstruction.revision import revise_problem_regions
from app.services.reconstruction.revision_integrity import assess_revision, inspect_assets, recover_legacy_revision_assets
from app.services.reconstruction.objectization_audit import audit_objectization
from app.services.reconstruction.white_objectization import objectize_on_white
from app.services.reconstruction.text_erasure import erase_editable_text_sources
from app.services.pptx import renderer
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


def _project(tmp_path: Path, *, suppressed: bool, duplicate: bool = False) -> tuple[ProjectStore, str, dict]:
    store = ProjectStore(tmp_path)
    project_id = store.create("revision-test")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((100, 240, 3), 255, np.uint8)
    cv2.putText(source, "HELLO", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    line = {"id": "text_001", "type": "text", "text": "HELLO", "x": 20, "y": 28, "width": 130, "height": 40, "zIndex": 2, "style": {"fontSize": 28, "fontFamily": "Arial", "color": "#000000"}, "metadata": {"rawOCRBBox": [20, 28, 135, 60], **({"suppressRender": True} if suppressed else {})}}
    elements = [line]
    if duplicate:
        second = {**line, "id": "text_002", "metadata": {"rawOCRBBox": [20, 28, 135, 60]}}
        elements.append(second)
    layout = {"version": "1.1", "slide": {"width": 240, "height": 100}, "elements": elements}
    store.save_slide(project_id, 1, layout)
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    preview = root / "reconstructed_preview.png"
    render_preview(root / "backgrounds" / "page_1.png", layout, preview)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    score.update({"detectedTextCount": 2 if duplicate else 1, "editableTextCoverage": 0 if suppressed else 1})
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    return store, project_id, layout


def test_revision_restores_missing_editable_text_without_full_analysis(tmp_path: Path) -> None:
    store, project_id, _ = _project(tmp_path, suppressed=True)
    result = revise_problem_regions(store, project_id, 1)
    assert result["revisionRound"] == 1
    assert result["accepted"] is True
    assert result["editableCoverageAfter"] > result["editableCoverageBefore"]
    assert any(issue["problem"] == "missingEditableText" for issue in result["issuesBefore"])
    assert store.get_slide(project_id, 1)["elements"][0]["metadata"].get("suppressRender") is None


def test_revision_keeps_previous_page_when_metrics_do_not_improve(tmp_path: Path) -> None:
    store, project_id, original = _project(tmp_path, suppressed=False, duplicate=True)
    result = revise_problem_regions(store, project_id, 1)
    assert result["revisionRound"] == 1
    assert result["accepted"] is False
    assert store.get_slide(project_id, 1) == original
    history = json.loads((tmp_path / project_id / "revision_history_1.json").read_text(encoding="utf-8"))
    assert history[0]["accepted"] is False


def test_revision_targets_missing_visual_before_color_mismatches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("missing-visual-priority")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((160, 300, 3), 255, np.uint8)
    cv2.rectangle(source, (190, 55), (240, 105), (35, 90, 185), -1)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    layout = {"slide": {"width": 300, "height": 160}, "elements": []}
    store.save_slide(project_id, 1, layout)
    preview = root / "reconstructed_preview.png"
    render_preview(root / "backgrounds" / "page_1.png", layout, preview)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    score["issues"].extend({"problem": "visualContentMismatch", "elementId": f"old_mismatch_{index}",
                            "bbox": [10 + index * 20, 15, 28 + index * 20, 35]} for index in range(4))
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    assert result["targetedIssues"][0]["problem"] == "missingVisualObject"
    assert any(issue["problem"] == "missingVisualObject" for issue in result["targetedIssues"])


def test_same_revision_round_on_two_pages_keeps_both_visual_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("two-page-assets")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    for page, color in ((1, (35, 90, 185)), (2, (80, 160, 45))):
        source = np.full((130, 240, 3), 255, np.uint8)
        cv2.rectangle(source, (70, 40), (110, 80), color, -1)
        source_path = root / ("source.png" if page == 1 else f"source_{page}.png")
        preview = root / ("reconstructed_preview.png" if page == 1 else f"reconstructed_preview_{page}.png")
        score_path = root / ("visual_score.json" if page == 1 else f"visual_score_{page}.json")
        background = root / "backgrounds" / f"page_{page}.png"
        cv2.imwrite(str(source_path), source)
        cv2.imwrite(str(background), np.full_like(source, 255))
        layout = {"slide": {"width": 240, "height": 130}, "elements": []}
        store.save_slide(project_id, page, layout)
        render_preview(background, layout, preview)
        score_path.write_text(json.dumps(run_visual_qa(source_path, preview, root, layout)), encoding="utf-8")

    first = revise_problem_regions(store, project_id, 1)
    first_image = next(item for item in store.get_slide(project_id, 1)["elements"] if item["type"] == "image")
    first_path = root / "assets" / Path(first_image["src"]).name
    first_bytes = first_path.read_bytes()
    second = revise_problem_regions(store, project_id, 2)
    second_image = next(item for item in store.get_slide(project_id, 2)["elements"] if item["type"] == "image")
    second_path = root / "assets" / Path(second_image["src"]).name

    assert first["accepted"] and second["accepted"]
    assert first["revisionRound"] == second["revisionRound"] == 1
    assert first_path != second_path
    assert first_path.read_bytes() == first_bytes
    assert second_path.is_file()
    assert inspect_assets(root, store.get_slide(project_id, 1))["missingAssetCount"] == 0
    assert inspect_assets(root, store.get_slide(project_id, 2))["missingAssetCount"] == 0


def test_revision_endpoint_uses_saved_page_and_reports_result(tmp_path: Path, monkeypatch) -> None:
    store, project_id, _ = _project(tmp_path, suppressed=True)
    monkeypatch.setattr(api_main, "store", store)
    response = TestClient(api_main.app).post(f"/api/projects/{project_id}/pages/1/revise")
    assert response.status_code == 200
    payload = response.json()
    assert payload["revisionRound"] == 1
    assert payload["accepted"] is True
    assert payload["layout"]["elements"][0]["text"] == "HELLO"
    assert TestClient(api_main.app).post(f"/api/projects/{project_id}/pages/2/revise").status_code == 404


def test_high_quality_loop_waits_for_consecutive_failed_rounds(tmp_path: Path, monkeypatch) -> None:
    store, project_id, layout = _project(tmp_path, suppressed=False)
    root = tmp_path / project_id
    (root / "conversion_report.json").write_text(json.dumps({"mode": "high_quality"}), encoding="utf-8")
    issue = {"problem": "wrongBBox", "elementId": "text_001"}
    outcomes = [True, False, False]

    def fake_round(*_args):
        accepted = outcomes.pop(0)
        return {"accepted": accepted, "layout": layout, "issuesBefore": [issue], "issuesAfter": [issue],
                "improvedRegions": ["text_001"] if accepted else [], "visualBefore": 0.5,
                "visualAfter": 0.6 if accepted else 0.5, "editableCoverageBefore": 1.0,
                "editableCoverageAfter": 1.0, "revisionRound": 3 - len(outcomes),
                "stagnationReason": None if accepted else "no_measurable_improvement"}

    monkeypatch.setattr(revision, "revise_problem_regions", fake_round)
    (root / "visual_score.json").write_text(json.dumps({"revisionStatus": "stagnated"}), encoding="utf-8")
    result = revision.run_revision_loop(store, project_id, 1)
    assert outcomes == []
    assert result["accepted"] is True
    assert result["stagnationReason"] == "consecutive_rounds_without_improvement"
    assert result["improvedRegions"] == ["text_001"]
    assert json.loads((root / "visual_score.json").read_text(encoding="utf-8"))["stagnationReason"] == result["stagnationReason"]


def test_accept_current_result_persists_decision(tmp_path: Path, monkeypatch) -> None:
    store, project_id, _ = _project(tmp_path, suppressed=False)
    monkeypatch.setattr(api_main, "store", store)
    response = TestClient(api_main.app).post(f"/api/projects/{project_id}/pages/1/accept-result")
    assert response.status_code == 200
    score = json.loads((tmp_path / project_id / "visual_score.json").read_text(encoding="utf-8"))
    assert score["revisionStatus"] == "user_accepted"


def _image_project(tmp_path: Path, monkeypatch) -> tuple[ProjectStore, str]:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(api_main, "OUTPUTS_DIR", tmp_path)
    store, project_id, layout = _project(tmp_path, suppressed=True)
    root = tmp_path / project_id
    (root / "assets").mkdir()
    image = np.full((38, 54, 3), (180, 40, 20), np.uint8)
    cv2.imwrite(str(root / "assets" / "visual.png"), image)
    layout["elements"].append({"id": "visual", "type": "image", "x": 160, "y": 20, "width": 54, "height": 38, "zIndex": 1,
                               "src": f"/media/assets/{project_id}/visual.png", "metadata": {"reconstructionStrategy": "cutout_image", "backgroundSeparated": True}})
    store.save_slide(project_id, 1, layout)
    render_preview(root / "backgrounds" / "page_1.png", layout, root / "reconstructed_preview.png")
    return store, project_id


def test_image_survives_two_revisions_and_media_url_stays_valid(tmp_path: Path, monkeypatch) -> None:
    store, project_id = _image_project(tmp_path, monkeypatch)
    root = tmp_path / project_id
    before = (root / "reconstructed_preview.png").read_bytes()
    first = revise_problem_regions(store, project_id, 1)
    second = revise_problem_regions(store, project_id, 1)
    current = store.get_slide(project_id, 1)
    assert first["accepted"] is True
    assert first["assetsBefore"] == first["assetsAfter"] == 1
    assert second["assetsAfter"] == 1
    assert second["missingAssetCount"] == 0
    assert inspect_assets(root, current)["missingAssetCount"] == 0
    assert next(item for item in current["elements"] if item["id"] == "visual")["src"] == f"/media/assets/{project_id}/visual.png"
    assert TestClient(api_main.app).get(f"/media/assets/{project_id}/visual.png").status_code == 200
    assert (root / "reconstructed_preview.png").is_file()
    assert (root / "reconstructed_preview.png").read_bytes() != before


def test_revision_localizes_image_from_another_project_before_asset_check(tmp_path: Path, monkeypatch) -> None:
    store, project_id = _image_project(tmp_path, monkeypatch)
    root = tmp_path / project_id
    external_id = store.create("previous-project")["id"]
    external_assets = tmp_path / external_id / "assets"
    external_assets.mkdir()
    source_asset = root / "assets" / "visual.png"
    external_asset = external_assets / "visual.png"
    external_asset.write_bytes(source_asset.read_bytes())
    layout = store.get_slide(project_id, 1)
    next(item for item in layout["elements"] if item["id"] == "visual")["src"] = f"/media/assets/{external_id}/visual.png"
    store.save_slide(project_id, 1, layout)

    result = revise_problem_regions(store, project_id, 1)
    current = store.get_slide(project_id, 1)
    image = next(item for item in current["elements"] if item["id"] == "visual")

    assert result["localizedAssetCount"] == 1
    assert image["src"].startswith(f"/media/assets/{project_id}/imported_")
    assert inspect_assets(root, current)["missingAssetCount"] == 0
    external_asset.unlink()
    assert inspect_assets(root, current)["missingAssetCount"] == 0


def test_complex_white_slide_keeps_visual_assets_across_two_revisions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    fixture = Path(__file__).parent / "fixtures" / "complex_modules.png"
    source = cv2.imread(str(fixture))
    assert source is not None
    store = ProjectStore(tmp_path)
    project_id = store.create("complex-revision")['id']
    root = tmp_path / project_id
    source_path = root / "source.png"
    background_path = root / "backgrounds" / "page_1.png"
    preview_path = root / "reconstructed_preview.png"
    cv2.imwrite(str(source_path), source)
    layout = {"slide": {"width": source.shape[1], "height": source.shape[0]}, "elements": []}
    objectize_on_white(source_path, background_path, layout, root / "assets", project_id, 1)
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(source_path)})
    store.save_slide(project_id, 1, layout)
    render_preview(background_path, layout, preview_path)
    score = run_visual_qa(source_path, preview_path, root, layout)
    score.update({"detectedTextCount": 0, "editableTextCoverage": 1.0})
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    for _ in range(2):
        prior = store.get_slide(project_id, 1)
        prior_preview = preview_path.read_bytes()
        prior_assets = inspect_assets(root, prior)["assets"]
        prior_audit = audit_objectization(source_path, background_path, preview_path, prior)
        result = revise_problem_regions(store, project_id, 1)
        current = store.get_slide(project_id, 1)
        current_audit = audit_objectization(source_path, background_path, preview_path, current)
        assert inspect_assets(root, current)["missingAssetCount"] == 0
        assert inspect_assets(root, current)["assets"] >= prior_assets
        assert current_audit["retainedVisualCoverage"] >= prior_audit["retainedVisualCoverage"] - 0.01
        assert current_audit["visualMismatchPixels"] <= prior_audit["visualMismatchPixels"] + max(24, round(prior_audit["salientVisualPixels"] * 0.005))
        if result["rollbackTriggered"]:
            assert current == prior
            assert preview_path.read_bytes() == prior_preview


def test_revision_ignores_stale_missing_visual_issue_when_badge_is_preserved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("preserved-badge")["id"]
    root = tmp_path / project_id
    (root / "assets").mkdir()
    (root / "backgrounds").mkdir()
    source = np.full((150, 240, 3), 255, np.uint8)
    cv2.circle(source, (95, 75), 40, (30, 85, 195), -1)
    cv2.line(source, (75, 75), (115, 75), (255, 255, 255), 6)
    cv2.line(source, (95, 55), (95, 95), (255, 255, 255), 6)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    badge = source[35:116, 55:136].copy()
    alpha = np.zeros((81, 81), np.uint8)
    cv2.circle(alpha, (40, 40), 40, 255, -1)
    cv2.imwrite(str(root / "assets" / "badge.png"), np.dstack((badge, alpha)))
    layout = {"slide": {"width": 240, "height": 150}, "elements": [
        {"id": "badge", "type": "image", "x": 55, "y": 35, "width": 81, "height": 81,
         "zIndex": 10, "src": f"/media/assets/{project_id}/badge.png",
         "metadata": {"reconstructionStrategy": "cutout_image", "wholeBadgeAsset": True}},
    ]}
    store.save_slide(project_id, 1, layout)
    preview = root / "reconstructed_preview.png"
    render_preview(root / "backgrounds" / "page_1.png", layout, preview)
    stale = {"problem": "missingVisualObject", "elementId": "unowned_75_55", "bbox": [75, 55, 116, 96], "pixelArea": 200}
    score = run_visual_qa(root / "source.png", preview, root, layout)
    score["issues"].append(stale)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    (root / "problem_report.json").write_text(json.dumps({"issuesAfter": [stale]}), encoding="utf-8")
    prior_preview = preview.read_bytes()

    result = revise_problem_regions(store, project_id, 1)

    assert not any(issue["problem"] == "missingVisualObject" for issue in result["targetedIssues"])
    assert inspect_assets(root, store.get_slide(project_id, 1))["assets"] == 1
    assert preview.read_bytes() == prior_preview


def test_scene_region_is_not_baked_into_verified_white_background() -> None:
    layout = {"slide": {"width": 300, "height": 180}, "elements": [
        {"id": "plate", "type": "rectangle", "x": 30, "y": 30, "width": 180, "height": 100,
         "style": {"fill": "#EAF2F8"}, "metadata": {"reconstructionStrategy": "native_shape"}},
    ]}
    scene = {"regions": [{"id": "region", "type": "figure", "confidence": 0.9,
                           "bbox": {"left": 30, "top": 30, "width": 180, "height": 100}}]}
    clear = {"objectizationAudit": {"whiteBackground": True, "backgroundResidualRegions": 0}}
    residual = {"objectizationAudit": {"whiteBackground": False, "backgroundResidualRegions": 1}}

    assert not any(issue["problem"] == "assetBakedIntoBackground"
                   for issue in revision.collect_revision_issues(layout, clear, scene))
    assert any(issue["problem"] == "assetBakedIntoBackground"
               for issue in revision.collect_revision_issues(layout, residual, scene))


def test_revision_restores_damaged_whole_badge_instead_of_adding_square_patch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("badge-revision")["id"]
    root = tmp_path / project_id
    (root / "assets").mkdir()
    (root / "backgrounds").mkdir()
    source = np.full((180, 240, 3), 255, np.uint8)
    cv2.circle(source, (110, 90), 50, (30, 40, 190), -1)
    cv2.putText(source, "AI", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    damaged = source[40:140, 60:160].copy()
    damaged[39:65, 31:71] = (245, 245, 245)
    alpha = np.zeros((100, 100), np.uint8)
    cv2.circle(alpha, (50, 50), 50, 255, -1)
    cv2.imwrite(str(root / "assets" / "badge.png"), np.dstack((damaged, alpha)))
    layout = {"slide": {"width": 240, "height": 180}, "elements": [
        {"id": "badge", "type": "image", "x": 60, "y": 40, "width": 100, "height": 100,
         "zIndex": 10, "src": f"/media/assets/{project_id}/badge.png",
         "metadata": {"reconstructionStrategy": "cutout_image", "preserveWholeAsset": True, "wholeBadgeAsset": True,
                      "textCleaned": True, "editableTextIds": ["letters"]}},
        {"id": "letters", "type": "text", "x": 90, "y": 79, "width": 40, "height": 26,
         "zIndex": 11, "text": "AI", "metadata": {"rawOCRBBox": [90, 79, 130, 105],
                                               "suppressed": True, "ownedBy": "badge"}},
    ]}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    store.save_slide(project_id, 1, layout)
    render_preview(root / "backgrounds" / "page_1.png", layout, root / "reconstructed_preview.png")
    score = run_visual_qa(root / "source.png", root / "reconstructed_preview.png", root, layout)
    score.update({"detectedTextCount": 1, "editableTextCoverage": 0.0})
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    before = audit_objectization(root / "source.png", root / "backgrounds" / "page_1.png", root / "reconstructed_preview.png", layout)
    assert any(issue["problem"] == "visualContentMismatch" for issue in before["issues"])

    result = revise_problem_regions(store, project_id, 1)

    current = store.get_slide(project_id, 1)
    after = audit_objectization(root / "source.png", root / "backgrounds" / "page_1.png", root / "reconstructed_preview.png", current)
    assert result["accepted"] is True
    assert result["replacedAssetCount"] == 1, {"targeted": result["targetedIssues"], "integrity": result["integrityErrors"], "changed": result["improvedRegions"]}
    assert result["missingAssetCount"] == 0
    assert after["visualMismatchPixels"] < before["visualMismatchPixels"] * 0.2
    assert len([item for item in current["elements"] if item["type"] == "image"]) == 1
    assert current["elements"][0]["metadata"]["sourceContentPreserved"] is True


def test_missing_candidate_asset_rolls_back_without_changing_preview(tmp_path: Path, monkeypatch) -> None:
    store, project_id = _image_project(tmp_path, monkeypatch)
    root = tmp_path / project_id
    baseline = store.get_slide(project_id, 1)
    preview = (root / "reconstructed_preview.png").read_bytes()

    def break_candidate(_background, candidate, **_kwargs):
        next(item for item in candidate["elements"] if item["type"] == "image")["src"] = f"/media/assets/{project_id}/missing.png"
        return {"backgroundTextErased": 0, "assetTextErased": 0}

    monkeypatch.setattr(revision, "erase_editable_text_sources", break_candidate)
    result = revise_problem_regions(store, project_id, 1)
    assert result["accepted"] is False
    assert result["rollbackTriggered"] is True
    assert result["candidateMissingAssetCount"] == 1
    assert result["missingAssetCount"] == 0
    assert store.get_slide(project_id, 1) == baseline
    assert (root / "reconstructed_preview.png").read_bytes() == preview


def test_white_background_regression_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    before = np.full((100, 100, 3), 120, np.uint8)
    after = np.full_like(before, 255)
    for name, array in (("before_bg.png", before), ("before_preview.png", before), ("after_bg.png", after), ("after_preview.png", after)):
        cv2.imwrite(str(root / name), array)
    result = assess_revision(root, {"elements": []}, {"elements": []}, root / "before_bg.png", root / "after_bg.png", root / "before_preview.png", root / "after_preview.png")
    assert "background_over_whitened" in result["integrityErrors"]
    assert "preview_over_whitened" in result["integrityErrors"]


def test_integrity_allows_only_declared_complete_image_replacement(tmp_path: Path) -> None:
    root = tmp_path / "project"
    assets = root / "assets"
    assets.mkdir(parents=True)
    for name in ("old.png", "new.png"):
        cv2.imwrite(str(assets / name), np.full((40, 50, 3), (35, 90, 185), np.uint8))
    white = np.full((100, 160, 3), 255, np.uint8)
    for name in ("before_bg.png", "after_bg.png", "before_preview.png", "after_preview.png"):
        cv2.imwrite(str(root / name), white)
    old = {"id": "old", "type": "image", "x": 30, "y": 25, "width": 50, "height": 40,
           "src": "/media/assets/project/old.png"}
    new = {"id": "new", "type": "image", "x": 29, "y": 24, "width": 52, "height": 42,
           "src": "/media/assets/project/new.png", "metadata": {"replacesAssetId": "old"}}
    baseline = {"elements": [old]}
    candidate = {"elements": [{**old, "metadata": {"suppressed": True, "replacedBy": "new"}}, new]}
    paths = [root / name for name in ("before_bg.png", "after_bg.png", "before_preview.png", "after_preview.png")]

    valid = assess_revision(root, baseline, candidate, *paths)
    assert valid["integrityErrors"] == []
    assert valid["replacedAssetCount"] == 1
    assert valid["assetsBefore"] == valid["assetsAfter"] == 1
    assert (assets / "old.png").is_file()

    incomplete = {"elements": [candidate["elements"][0], {**new, "x": 62, "width": 19}]}
    assert "lost_existing_images" in assess_revision(root, baseline, incomplete, *paths)["integrityErrors"]
    unlinked = {"elements": [old | {"metadata": {"suppressed": True}}, new]}
    assert "lost_existing_images" in assess_revision(root, baseline, unlinked, *paths)["integrityErrors"]


def test_revision_replaces_wrong_color_image_without_deleting_original_asset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("wrong-color-image")["id"]
    root = tmp_path / project_id
    (root / "assets").mkdir()
    (root / "backgrounds").mkdir()
    source = np.full((110, 170, 3), 255, np.uint8)
    cv2.rectangle(source, (45, 35), (94, 74), (30, 100, 190), -1)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    old_asset = root / "assets" / "old.png"
    cv2.imwrite(str(old_asset), np.full((40, 50, 3), (160, 180, 40), np.uint8))
    layout = {"slide": {"width": 170, "height": 110}, "elements": [
        {"id": "old", "type": "image", "x": 45, "y": 35, "width": 50, "height": 40,
         "zIndex": 10, "src": f"/media/assets/{project_id}/old.png",
         "metadata": {"reconstructionStrategy": "cutout_image"}},
    ]}
    store.save_slide(project_id, 1, layout)
    preview = root / "reconstructed_preview.png"
    render_preview(root / "backgrounds" / "page_1.png", layout, preview)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    current = store.get_slide(project_id, 1)
    assert result["accepted"] is True
    assert result["replacedAssetCount"] == 1
    assert result["missingAssetCount"] == 0
    assert old_asset.is_file()
    assert next(item for item in current["elements"] if item["id"] == "old")["metadata"]["suppressed"] is True
    assert np.array_equal(cv2.imread(str(preview))[50, 60], source[50, 60])


def test_local_visual_loss_is_rejected_even_below_page_white_threshold() -> None:
    assert revision._visual_retention_regressed(
        {"retainedVisualCoverage": 0.98, "missingVisualPixels": 10},
        {"retainedVisualCoverage": 0.91, "missingVisualPixels": 48},
    )
    assert not revision._visual_retention_regressed(
        {"retainedVisualCoverage": 0.98, "missingVisualPixels": 10},
        {"retainedVisualCoverage": 0.981, "missingVisualPixels": 9},
    )
    assert revision._visual_retention_regressed(
        {"retainedVisualCoverage": 0.9998, "missingVisualPixels": 20, "salientVisualPixels": 500000, "largestMissingVisualRegion": 8},
        {"retainedVisualCoverage": 0.9996, "missingVisualPixels": 140, "salientVisualPixels": 500000, "largestMissingVisualRegion": 120},
    )
    assert not revision._visual_retention_regressed(
        {"retainedVisualCoverage": 0.9998, "missingVisualPixels": 20, "salientVisualPixels": 500000, "largestMissingVisualRegion": 8},
        {"retainedVisualCoverage": 0.9998, "missingVisualPixels": 30, "salientVisualPixels": 500000, "largestMissingVisualRegion": 9},
    )
    assert not revision._visual_retention_regressed(
        {"missingVisualPixels": 6, "salientVisualPixels": 600000, "largestMissingVisualRegion": 3},
        {"missingVisualPixels": 456, "salientVisualPixels": 600000, "largestMissingVisualRegion": 25},
    )
    assert revision._visual_retention_regressed(
        {"retainedVisualCoverage": 0.98, "missingVisualPixels": 10, "salientVisualPixels": 10000, "visualMismatchPixels": 120},
        {"retainedVisualCoverage": 0.98, "missingVisualPixels": 10, "salientVisualPixels": 10000, "visualMismatchPixels": 190},
    )
    assert not revision._visual_retention_regressed(
        {"salientVisualPixels": 10000, "visualMismatchPixels": 120},
        {"salientVisualPixels": 10000, "visualMismatchPixels": 135},
    )


def test_unrelated_preview_region_change_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    before = np.full((100, 100, 3), 100, np.uint8)
    after = before.copy()
    after[50:90, 50:90] = 180
    cv2.imwrite(str(root / "before.png"), before)
    cv2.imwrite(str(root / "after.png"), after)
    result = assess_revision(root, {"elements": []}, {"elements": []}, root / "before.png", root / "before.png", root / "before.png", root / "after.png", [[0, 0, 20, 20]])
    assert "untargeted_preview_change" in result["integrityErrors"]


def test_legacy_revision_asset_url_can_be_recovered(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "assets").mkdir(parents=True)
    (root / "revisions" / "assets").mkdir(parents=True)
    cv2.imwrite(str(root / "assets" / "visual.png"), np.full((8, 8, 3), 100, np.uint8))
    cv2.imwrite(str(root / "revisions" / "assets" / "revision_2_visual.png"), np.full((8, 8, 3), 255, np.uint8))
    layout = {"elements": [{"id": "visual", "type": "image", "src": "/media/assets/revisions/revision_2_visual.png"}]}
    assert recover_legacy_revision_assets(root, layout) == 1
    assert layout["elements"][0]["src"] == "/media/assets/project/visual.png"
    assert inspect_assets(root, layout)["missingAssetCount"] == 0


def test_revision_text_cleaning_stages_valid_project_asset_and_protects_background(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    root = tmp_path / "project"
    (root / "assets").mkdir(parents=True)
    candidate_dir = root / "revisions" / "round_1"
    candidate_dir.mkdir(parents=True)
    background = np.full((80, 120, 3), (50, 70, 90), np.uint8)
    cv2.imwrite(str(candidate_dir / "background.png"), background)
    asset = np.full((40, 80, 3), (20, 50, 180), np.uint8)
    cv2.putText(asset, "HELLO", (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(root / "assets" / "visual.png"), asset)
    layout = {"elements": [
        {"id": "visual", "type": "image", "x": 10, "y": 10, "width": 80, "height": 40, "src": "/media/assets/project/visual.png"},
        {"id": "text", "type": "text", "text": "HELLO", "x": 12, "y": 12, "width": 60, "height": 25, "metadata": {"rawOCRBBox": [12, 12, 72, 37]}},
    ]}
    erase_editable_text_sources(candidate_dir / "background.png", layout, target_text_ids={"text"}, copy_asset_prefix="revision_1", project_root=root, protect_background_elements=True)
    assert layout["elements"][0]["src"].startswith("/media/assets/project/revision_1_")
    assert inspect_assets(root, layout)["missingAssetCount"] == 0
    assert cv2.imread(str(candidate_dir / "background.png")).tolist() == background.tolist()
    assert cv2.imread(str(root / "assets" / "visual.png")).tolist() == asset.tolist()


def test_commit_failure_restores_previous_preview_and_score(tmp_path: Path, monkeypatch) -> None:
    store, project_id, layout = _project(tmp_path, suppressed=False)
    root = tmp_path / project_id
    workspace = root / "revisions" / "round_1"
    workspace.mkdir(parents=True)
    preview = root / "reconstructed_preview.png"
    original_preview = preview.read_bytes()
    original_score = (root / "visual_score.json").read_bytes()
    changed = workspace / "changed.png"
    cv2.imwrite(str(changed), np.zeros((100, 240, 3), np.uint8))

    def fail_save(*_args):
        raise OSError("simulated write failure")

    monkeypatch.setattr(store, "save_slide", fail_save)
    with pytest.raises(OSError, match="simulated"):
        revision._commit_revision(store, project_id, 1, layout, workspace, {preview: changed}, {root / "visual_score.json": {"overall": 0}})
    assert preview.read_bytes() == original_preview
    assert (root / "visual_score.json").read_bytes() == original_score
