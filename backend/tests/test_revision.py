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


def test_revision_rolls_back_when_new_text_ghosting_appears(tmp_path: Path, monkeypatch) -> None:
    store, project_id, baseline = _project(tmp_path, suppressed=True)
    original_preview = (tmp_path / project_id / "reconstructed_preview.png").read_bytes()
    monkeypatch.setattr(revision, "count_text_ghosting", lambda _source, background, _layout: int("revisions" in Path(background).parts))

    result = revise_problem_regions(store, project_id, 1)

    assert result["accepted"] is False
    assert result["rollbackTriggered"] is True
    assert "text_ghosting_regressed" in result["integrityErrors"]
    assert store.get_slide(project_id, 1) == baseline
    assert (tmp_path / project_id / "reconstructed_preview.png").read_bytes() == original_preview


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


def test_rejected_revision_discards_only_new_candidate_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("rollback-assets")["id"]
    root = tmp_path / project_id
    (root / "assets").mkdir()
    (root / "backgrounds").mkdir()
    existing = root / "assets" / "existing.png"
    cv2.imwrite(str(existing), np.full((12, 12, 3), (35, 95, 185), np.uint8))
    existing_bytes = existing.read_bytes()
    source = np.full((130, 240, 3), 255, np.uint8)
    cv2.rectangle(source, (80, 40), (120, 80), (45, 100, 190), -1)
    cv2.imwrite(str(root / "source.png"), source)
    background = root / "backgrounds" / "page_1.png"
    cv2.imwrite(str(background), np.full_like(source, 255))
    layout = {"slide": {"width": 240, "height": 130}, "elements": []}
    store.save_slide(project_id, 1, layout)
    preview = root / "reconstructed_preview.png"
    render_preview(background, layout, preview)
    preview_bytes = preview.read_bytes()
    (root / "visual_score.json").write_text(json.dumps(run_visual_qa(root / "source.png", preview, root, layout)), encoding="utf-8")
    real_assess = revision.assess_revision

    def reject_candidate(*args, **kwargs):
        result = real_assess(*args, **kwargs)
        result["integrityErrors"].append("forced_rejection")
        return result

    monkeypatch.setattr(revision, "assess_revision", reject_candidate)
    result = revise_problem_regions(store, project_id, 1)

    assert result["accepted"] is False
    assert result["rollbackTriggered"] is True
    assert result["discardedCandidateAssets"] >= 1
    assert list((root / "assets").glob("*.png")) == [existing]
    assert existing.read_bytes() == existing_bytes
    assert preview.read_bytes() == preview_bytes


def test_revision_endpoint_uses_saved_page_and_reports_result(tmp_path: Path, monkeypatch) -> None:
    store, project_id, _ = _project(tmp_path, suppressed=True)
    monkeypatch.setattr(api_main, "store", store)
    response = TestClient(api_main.app).post(f"/api/projects/{project_id}/pages/1/revise")
    assert response.status_code == 200
    payload = response.json()
    assert payload["revisionRound"] == 1
    assert payload["accepted"] is True
    assert payload["newlyLostVisualPixels"] == 0
    assert payload["largestNewVisualLoss"] == 0
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


def test_audited_baked_visual_reaches_revision_without_scene_regions() -> None:
    layout = {"slide": {"width": 400, "height": 200}, "elements": []}
    score = {"issues": [{"problem": "assetBakedIntoBackground", "elementId": "background_20_20",
                         "bbox": [20, 20, 380, 180], "pixelArea": 57600}],
             "objectizationAudit": {"whiteBackground": False, "backgroundResidualRegions": 1}}

    issues = revision.collect_revision_issues(layout, score)

    assert any(issue["problem"] == "assetBakedIntoBackground"
               and issue["bbox"] == [20, 20, 380, 180] for issue in issues)


@pytest.mark.parametrize("visual_box", [(20, 20, 380, 180), (10, 10, 390, 190)])
def test_revision_objectizes_large_visual_baked_into_background(tmp_path: Path, monkeypatch, visual_box) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("baked-large-visual")["id"]
    root = tmp_path / project_id
    (root / "assets").mkdir()
    (root / "backgrounds").mkdir()
    source = np.full((200, 400, 3), 255, np.uint8)
    x1, y1, x2, y2 = visual_box
    source[y1:y2, x1:x2] = (40, 110, 190)
    cv2.imwrite(str(root / "source.png"), source)
    background = root / "backgrounds" / "page_1.png"
    cv2.imwrite(str(background), source)
    preview = root / "reconstructed_preview.png"
    cv2.imwrite(str(preview), source)
    layout = {"slide": {"width": 400, "height": 200}, "elements": []}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    store.save_slide(project_id, 1, layout)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    current = store.get_slide(project_id, 1)
    assert result["accepted"] is True, result
    assert result["missingAssetCount"] == 0
    assert any(item["type"] == "image" for item in current["elements"])
    assert np.all(cv2.imread(str(background))[40:160, 40:360] >= 250)
    assert np.max(np.abs(cv2.imread(str(preview))[100, 100].astype(int) - source[100, 100].astype(int))) < 5


def test_revision_moves_flat_page_surface_out_of_legacy_background(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("baked-page-surface")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((180, 300, 3), (52, 31, 21), np.uint8)
    cv2.imwrite(str(root / "source.png"), source)
    background = root / "backgrounds" / "page_1.png"
    cv2.imwrite(str(background), source)
    preview = root / "reconstructed_preview.png"
    cv2.imwrite(str(preview), source)
    layout = {"slide": {"width": 300, "height": 180}, "elements": []}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    store.save_slide(project_id, 1, layout)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    current = store.get_slide(project_id, 1)
    assert result["accepted"] is True, result
    assert any((item.get("metadata") or {}).get("pageSurface") for item in current["elements"])
    assert np.all(cv2.imread(str(background)) == 255)
    assert np.array_equal(cv2.imread(str(preview))[90, 150], source[90, 150])
    assert not audit_objectization(root / "source.png", background, preview, current)["issues"]


def test_revision_objectizes_unowned_visual_on_legacy_page_surface(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("surface-with-unowned-mark")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((180, 300, 3), (52, 31, 21), np.uint8)
    cv2.circle(source, (150, 90), 22, (245, 245, 245), -1)
    cv2.imwrite(str(root / "source.png"), source)
    background = root / "backgrounds" / "page_1.png"
    cv2.imwrite(str(background), source)
    preview = root / "reconstructed_preview.png"
    cv2.imwrite(str(preview), source)
    layout = {"slide": {"width": 300, "height": 180}, "elements": []}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    store.save_slide(project_id, 1, layout)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    current = store.get_slide(project_id, 1)
    assert result["accepted"] is True, result
    assert result["rollbackTriggered"] is False
    assert result["missingAssetCount"] == 0
    assert any(item["type"] == "image" and item["x"] <= 150 < item["x"] + item["width"]
               for item in current["elements"])
    assert np.all(cv2.imread(str(background)) == 255)
    assert np.array_equal(cv2.imread(str(preview))[90, 150], source[90, 150])


def test_surface_residual_occupancy_keeps_support_inside_large_textbox() -> None:
    source = np.full((180, 300, 3), (52, 31, 21), np.uint8)
    cv2.putText(source, "T", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 245, 245), 2)
    cv2.circle(source, (150, 90), 22, (245, 245, 245), -1)
    layout = {"elements": [{"id": "label", "type": "text", "text": "T", "x": 20, "y": 20,
                            "width": 240, "height": 120,
                            "metadata": {"rawOCRBBox": [20, 25, 55, 65]}}]}

    occupied = revision._surface_residual_occupancy(source, layout, "page_surface_1")

    assert np.count_nonzero(occupied[25:65, 20:55]) > 0
    assert occupied[90, 150] == 0


def test_revision_preserves_icon_within_editable_textbox_on_legacy_surface(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("surface-text-and-icon")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((180, 300, 3), (52, 31, 21), np.uint8)
    cv2.putText(source, "T", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 245, 245), 2)
    cv2.circle(source, (150, 90), 22, (245, 245, 245), -1)
    cv2.imwrite(str(root / "source.png"), source)
    background = root / "backgrounds" / "page_1.png"
    cv2.imwrite(str(background), source)
    preview = root / "reconstructed_preview.png"
    cv2.imwrite(str(preview), source)
    layout = {"slide": {"width": 300, "height": 180}, "elements": [
        {"id": "label", "type": "text", "text": "T", "x": 20, "y": 25, "width": 240, "height": 100,
         "zIndex": 20, "style": {"color": "#F5F5F5", "fontSize": 30},
         "metadata": {"rawOCRBBox": [20, 25, 55, 65]}}]}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    store.save_slide(project_id, 1, layout)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    current = store.get_slide(project_id, 1)
    assert result["accepted"] is True, result
    assert any(item["id"] == "label" and item["type"] == "text" for item in current["elements"])
    assert any(item["type"] == "image" and item["x"] <= 150 < item["x"] + item["width"]
               for item in current["elements"])
    assert np.all(cv2.imread(str(background)) == 255)
    assert np.array_equal(cv2.imread(str(preview))[90, 150], source[90, 150])


def test_revision_restores_flat_dark_page_without_flattening_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("dark-surface")['id']
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((180, 300, 3), (52, 31, 21), np.uint8)
    cv2.putText(source, "Dark", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 245, 245), 2)
    cv2.imwrite(str(root / "source.png"), source)
    background = root / "backgrounds" / "page_1.png"
    cv2.imwrite(str(background), np.full_like(source, 255))
    layout = {"slide": {"width": 300, "height": 180}, "elements": [
        {"id": "background_001", "type": "background", "x": 0, "y": 0, "width": 300, "height": 180,
         "zIndex": 0, "src": f"/media/backgrounds/{project_id}/page_1.png"},
        {"id": "title", "type": "text", "x": 35, "y": 45, "width": 100, "height": 45,
         "zIndex": 20, "text": "Dark", "style": {"color": "#f5f5f5", "fontSize": 32},
         "metadata": {"rawOCRBBox": [35, 45, 135, 90]}}]}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    store.save_slide(project_id, 1, layout)
    preview = root / "reconstructed_preview.png"
    render_preview(background, layout, preview)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    assert result["accepted"] is True, result.get("integrityErrors")
    assert result["rollbackTriggered"] is False
    assert any(issue["problem"] == "pageSurfaceLost" for issue in result["issuesBefore"])
    assert not any(issue["problem"] == "pageSurfaceLost" for issue in result["issuesAfter"])
    current = store.get_slide(project_id, 1)
    assert len([item for item in current["elements"] if (item.get("metadata") or {}).get("pageSurface")]) == 1
    assert cv2.imread(str(preview))[150, 150].tolist() == [52, 31, 21]


def test_revision_restores_large_textured_visual_without_whitening(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("large-texture-revision")["id"]
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((200, 400, 3), 255, np.uint8)
    for y in range(20, 180):
        source[y, 20:380] = (50 + y // 4, 110, 190)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    layout = {"slide": {"width": 400, "height": 200}, "elements": []}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    store.save_slide(project_id, 1, layout)
    preview = root / "reconstructed_preview.png"
    render_preview(root / "backgrounds" / "page_1.png", layout, preview)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    assert result["accepted"] is True, result.get("integrityErrors")
    assert any(issue["problem"] == "largeVisualLoss" for issue in result["issuesBefore"])
    assert result["missingAssetCount"] == 0
    assert not result["rollbackTriggered"]
    assert any(item["type"] == "image" for item in store.get_slide(project_id, 1)["elements"])
    assert cv2.imread(str(preview))[100, 100].tolist() == source[100, 100].tolist()


def test_revision_replaces_large_wrong_color_asset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("large-mismatch")['id']
    root = tmp_path / project_id
    (root / "assets").mkdir()
    (root / "backgrounds").mkdir()
    source = np.full((200, 400, 3), 255, np.uint8)
    source[20:180, 20:380] = (40, 110, 190)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    wrong = np.full((160, 360, 3), (190, 110, 40), np.uint8)
    cv2.imwrite(str(root / "assets" / "old.png"), wrong)
    layout = {"slide": {"width": 400, "height": 200}, "elements": [
        {"id": "old", "type": "image", "x": 20, "y": 20, "width": 360, "height": 160,
         "zIndex": 10, "src": f"/media/assets/{project_id}/old.png"}]}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    store.save_slide(project_id, 1, layout)
    preview = root / "reconstructed_preview.png"
    render_preview(root / "backgrounds" / "page_1.png", layout, preview)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    before = audit_objectization(root / "source.png", root / "backgrounds" / "page_1.png", preview, layout)

    result = revise_problem_regions(store, project_id, 1)

    current = store.get_slide(project_id, 1)
    after = audit_objectization(root / "source.png", root / "backgrounds" / "page_1.png", preview, current)
    assert result["accepted"] is True, result
    assert result["replacedAssetCount"] == 1
    assert result["missingAssetCount"] == 0
    assert (root / "assets" / "old.png").exists()
    assert after["visualMismatchPixels"] < before["visualMismatchPixels"] * 0.2


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


@pytest.mark.parametrize("damage_kind", ["dark_bleed", "pale_fade"])
def test_revision_repairs_residual_bleed_without_overwriting_previous_asset(tmp_path: Path, monkeypatch,
                                                                            damage_kind: str) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("pale-gap-revision")["id"]
    root = tmp_path / project_id
    (root / "assets").mkdir()
    (root / "backgrounds").mkdir()
    source = np.full((180, 320, 3), 255, np.uint8)
    source[30:63, 30:250] = (30, 50, 80)
    if damage_kind == "pale_fade":
        source[63:100, 30:250] = (225, 230, 253)
    damaged = source[30:100, 30:250].copy()
    damaged[33:48, 10:170] = (30, 50, 80) if damage_kind == "dark_bleed" else (253, 254, 252)
    source_path = root / "source.png"
    old_asset = root / "assets" / "residual.png"
    background = root / "backgrounds" / "page_1.png"
    preview = root / "reconstructed_preview.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(background), np.full_like(source, 255))
    cv2.imwrite(str(old_asset), np.dstack((damaged, np.full(damaged.shape[:2], 255, np.uint8))))
    old_bytes = old_asset.read_bytes()
    layout = {"slide": {"width": 320, "height": 180}, "elements": [
        {"id": "residual", "type": "image", "x": 30, "y": 30, "width": 220, "height": 70,
         "zIndex": 2, "src": f"/media/assets/{project_id}/residual.png",
         "metadata": {"layerRole": "residual", "reconstructionStrategy": "cutout_image"}},
    ]}
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(source_path)})
    store.save_slide(project_id, 1, layout)
    render_preview(background, layout, preview)
    score = run_visual_qa(source_path, preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")
    before = audit_objectization(source_path, background, preview, layout)
    metric = "falseVisualAdditionPixels" if damage_kind == "dark_bleed" else "fadedPaleSupportPixels"
    assert before[metric] > 1000

    result = revise_problem_regions(store, project_id, 1)

    current = store.get_slide(project_id, 1)
    after = audit_objectization(source_path, background, preview, current)
    assert result["accepted"] is True, result
    assert result["missingAssetCount"] == 0
    assert after[metric] == 0
    assert old_asset.read_bytes() == old_bytes
    assert current["elements"][0]["src"] != layout["elements"][0]["src"]


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


def test_integrity_rejects_silent_shape_loss_even_when_preview_is_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    white = np.full((100, 160, 3), 255, np.uint8)
    for name in ("before_bg.png", "after_bg.png", "before_preview.png", "after_preview.png"):
        cv2.imwrite(str(root / name), white)
    old = {"id": "plate", "type": "rectangle", "x": 30, "y": 25, "width": 80, "height": 40,
           "style": {"fill": "#EAF2F8"}}
    paths = [root / name for name in ("before_bg.png", "after_bg.png", "before_preview.png", "after_preview.png")]
    baseline = {"elements": [old]}

    lost = {"elements": [{**old, "metadata": {"suppressed": True}}]}
    assert "lost_existing_shapes" in assess_revision(root, baseline, lost, *paths)["integrityErrors"]

    replacement = {"id": "plate_cutout", "type": "rectangle", "x": 29, "y": 24,
                   "width": 82, "height": 42, "style": {"fill": "#EAF2F8"}}
    candidate = {"elements": [lost["elements"][0], replacement]}
    assert "lost_existing_shapes" not in assess_revision(root, baseline, candidate, *paths)["integrityErrors"]


def test_revision_keeps_monolithic_image_issue_until_it_is_actually_fixed() -> None:
    issue = {"problem": "monolithicPageImage", "elementId": "screenshot", "bbox": [0, 0, 300, 180]}
    assert issue in revision.collect_revision_issues({"elements": []}, {"issues": [issue]})


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
        {"monolithicPageImageCount": 0}, {"monolithicPageImageCount": 1})
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
    assert revision._visual_retention_regressed(
        {"missingVisualPixels": 6, "salientVisualPixels": 600000, "largestMissingVisualRegion": 3},
        {"missingVisualPixels": 456, "salientVisualPixels": 600000, "largestMissingVisualRegion": 25},
    )
    assert not revision._visual_retention_regressed(
        {"missingVisualPixels": 6, "salientVisualPixels": 600000, "largestMissingVisualRegion": 3},
        {"missingVisualPixels": 80, "salientVisualPixels": 600000, "largestMissingVisualRegion": 10},
    )
    assert revision._visual_retention_regressed(
        {"retainedVisualCoverage": 0.98, "missingVisualPixels": 10, "salientVisualPixels": 10000, "visualMismatchPixels": 120},
        {"retainedVisualCoverage": 0.98, "missingVisualPixels": 10, "salientVisualPixels": 10000, "visualMismatchPixels": 190},
    )
    assert not revision._visual_retention_regressed(
        {"salientVisualPixels": 10000, "visualMismatchPixels": 120},
        {"salientVisualPixels": 10000, "visualMismatchPixels": 135},
    )
    assert revision._visual_retention_regressed(
        {"salientVisualPixels": 600000, "visualMismatchPixels": 200},
        {"salientVisualPixels": 600000, "visualMismatchPixels": 410},
    )
    assert revision._visual_retention_regressed(
        {"missingVisualPixels": 10, "paleAssetGapPixels": 200},
        {"missingVisualPixels": 10, "paleAssetGapPixels": 320},
    )
    assert not revision._visual_retention_regressed(
        {"missingVisualPixels": 10, "paleAssetGapPixels": 200},
        {"missingVisualPixels": 10, "paleAssetGapPixels": 220},
    )


def test_revision_guard_rejects_many_small_deleted_decorations(tmp_path: Path) -> None:
    source = np.full((300, 500, 3), 255, np.uint8)
    for row in range(5):
        for column in range(8):
            x, y = 30 + column * 35, 35 + row * 35
            source[y:y + 4, x:x + 4] = (30, 60, 190)
    background = np.full_like(source, 255)
    cv2.imwrite(str(tmp_path / "source.png"), source)
    cv2.imwrite(str(tmp_path / "background.png"), background)
    cv2.imwrite(str(tmp_path / "before.png"), source)
    cv2.imwrite(str(tmp_path / "after.png"), background)
    before = audit_objectization(tmp_path / "source.png", tmp_path / "background.png",
                                 tmp_path / "before.png", {"elements": []})
    after = audit_objectization(tmp_path / "source.png", tmp_path / "background.png",
                                tmp_path / "after.png", {"elements": []})
    assert after["largestMissingVisualRegion"] < 24
    assert after["missingVisualPixels"] >= 500
    assert revision._visual_retention_regressed(before, after)


def test_revision_rejects_new_icon_loss_hidden_by_larger_visual_repair(tmp_path: Path) -> None:
    source = np.full((180, 300, 3), 255, np.uint8)
    cv2.rectangle(source, (25, 35), (105, 85), (70, 125, 190), -1)
    cv2.circle(source, (245, 125), 9, (35, 90, 190), -1)
    before = source.copy()
    before[35:86, 25:106] = 255  # Large plate is missing in the first revision.
    after = source.copy()
    after[116:135, 236:255] = 255  # Repair plate, accidentally erase small icon.
    for name, image in (("source.png", source), ("before.png", before), ("after.png", after),
                        ("background.png", np.full_like(source, 255))):
        cv2.imwrite(str(tmp_path / name), image)
    baseline_audit = audit_objectization(tmp_path / "source.png", tmp_path / "background.png",
                                         tmp_path / "before.png", {"elements": []})
    candidate_audit = audit_objectization(tmp_path / "source.png", tmp_path / "background.png",
                                          tmp_path / "after.png", {"elements": []})
    assert candidate_audit["missingVisualPixels"] < baseline_audit["missingVisualPixels"]
    integrity = assess_revision(tmp_path, {"elements": []}, {"elements": []},
                                tmp_path / "background.png", tmp_path / "background.png",
                                tmp_path / "before.png", tmp_path / "after.png", [[0, 0, 300, 180]],
                                source_path=tmp_path / "source.png")
    assert integrity["newlyLostVisualPixels"] >= 200
    assert "new_source_visual_loss" in integrity["integrityErrors"]


def test_revision_guard_keeps_colored_badge_inside_editable_text_bounds(tmp_path: Path) -> None:
    source = np.full((120, 220, 3), 255, np.uint8)
    cv2.putText(source, "INFO", (25, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (15, 15, 15), 2)
    cv2.circle(source, (130, 55), 11, (40, 120, 210), -1)
    after = source.copy()
    cv2.circle(after, (130, 55), 11, (255, 255, 255), -1)
    for name, image in (("source.png", source), ("before.png", source), ("after.png", after),
                        ("background.png", np.full_like(source, 255))):
        cv2.imwrite(str(tmp_path / name), image)
    layout = {"elements": [{"id": "info", "type": "text", "x": 20, "y": 28, "width": 130, "height": 48,
                            "style": {"color": "#0F0F0F"},
                            "metadata": {"rawOCRBBox": [20, 28, 150, 76]}}]}
    result = assess_revision(tmp_path, layout, layout, tmp_path / "background.png",
                             tmp_path / "background.png", tmp_path / "before.png", tmp_path / "after.png",
                             [[20, 28, 150, 76]], source_path=tmp_path / "source.png")
    assert result["largestNewVisualLoss"] >= 300
    assert "new_source_visual_loss" in result["integrityErrors"]


def test_revision_replaces_false_plate_with_source_visual_asset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(renderer, "OUTPUTS_DIR", tmp_path)
    store = ProjectStore(tmp_path)
    project_id = store.create("false-plate")['id']
    root = tmp_path / project_id
    (root / "backgrounds").mkdir()
    source = np.full((240, 400, 3), 255, np.uint8)
    skyline = np.array([[95, 165], [95, 145], [120, 145], [120, 110], [145, 110],
                        [145, 135], [180, 135], [180, 95], [210, 95], [210, 145],
                        [245, 145], [245, 165]], np.int32)
    cv2.fillPoly(source, [skyline], (238, 232, 251))
    cv2.rectangle(source, (122, 118), (139, 132), (252, 252, 252), -1)
    cv2.imwrite(str(root / "source.png"), source)
    cv2.imwrite(str(root / "backgrounds" / "page_1.png"), np.full_like(source, 255))
    layout = {"version": "1.1", "slide": {"width": 400, "height": 240}, "elements": [
        {"id": "false_plate", "type": "roundedRectangle", "x": 90, "y": 75,
         "width": 170, "height": 100, "zIndex": 10,
         "style": {"fill": "#FBD9D8", "stroke": "#FBD9D8", "strokeWidth": 1}},
    ]}
    store.save_slide(project_id, 1, layout)
    store.add_image(project_id, {"id": "source", "name": "source.png", "path": str(root / "source.png")})
    preview = root / "reconstructed_preview.png"
    render_preview(root / "backgrounds" / "page_1.png", layout, preview)
    score = run_visual_qa(root / "source.png", preview, root, layout)
    (root / "visual_score.json").write_text(json.dumps(score), encoding="utf-8")

    result = revise_problem_regions(store, project_id, 1)

    assert any(issue["problem"] == "unsupportedNativeShape" for issue in result["issuesBefore"])
    assert result["accepted"] is True
    current = store.get_slide(project_id, 1)
    assert current["elements"][0]["metadata"]["suppressed"] is True
    assert len([item for item in current["elements"] if item["type"] == "image"]) == 1
    assert np.array_equal(cv2.imread(str(preview))[125, 130], source[125, 130])
    assert np.all(cv2.imread(str(preview))[80, 100] == 255)


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
