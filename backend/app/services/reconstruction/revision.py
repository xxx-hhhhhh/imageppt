from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from app.config import OUTPUTS_DIR
from app.models.project_store import ProjectStore
from app.services.reconstruction.layered_background import separate_foreground
from app.services.reconstruction.pipeline import ReconstructionPipeline
from app.services.reconstruction.planner import AIReconstructionPlanner
from app.services.reconstruction.text_coverage import fit_text_to_ocr_lines, measure_text_coverage
from app.services.reconstruction.text_erasure import erase_editable_text_sources
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


def revise_problem_regions(store: ProjectStore, project_id: str, page: int) -> dict:
    """Attempt a regional revision using saved slide and QA artifacts only."""
    root = store.root / project_id
    source = root / ("source.png" if page == 1 else f"source_{page}.png")
    background = root / "backgrounds" / f"page_{page}.png"
    preview = root / ("reconstructed_preview.png" if page == 1 else f"reconstructed_preview_{page}.png")
    score_path = root / ("visual_score.json" if page == 1 else f"visual_score_{page}.json")
    if not all(path.is_file() for path in (source, background, preview, score_path)):
        raise FileNotFoundError("Page analysis artifacts are not available")
    baseline = store.get_slide(project_id, page)
    score_before = json.loads(score_path.read_text(encoding="utf-8"))
    history_path = root / f"revision_history_{page}.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    round_number = len(history) + 1
    scene_path = root / ("scene_raw.json" if page == 1 else f"scene_raw_{page}.json")
    raw_scene = json.loads(scene_path.read_text(encoding="utf-8")) if scene_path.is_file() else {}
    issues_before = collect_revision_issues(baseline, score_before, raw_scene)
    priority = {"ghosting": 0, "duplicateText": 1, "wrongOwnership": 2, "wrongBBox": 3, "textOverlap": 4, "missingEditableText": 5, "brokenChartOrModule": 6, "assetBakedIntoBackground": 7}
    tried = {(item.get("problem"), item.get("elementId")) for attempt in history if not attempt.get("accepted") for item in attempt.get("targetedIssues", [])}
    ranked = sorted(issues_before, key=lambda item: priority.get(item["problem"], 9))
    target_issues = [item for item in ranked if (item.get("problem"), item.get("elementId")) not in tried][:4]
    candidate = copy.deepcopy(baseline)
    candidate_dir = root / "revisions" / f"round_{round_number}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_bg = candidate_dir / "background.png"
    candidate_preview = candidate_dir / "preview.png"
    shutil.copy2(background, candidate_bg)
    touched_text: set[str] = set()
    erase_text: set[str] = set()
    changed_ids: set[str] = set()
    by_id = {str(item.get("id")): item for item in candidate.get("elements", [])}

    for issue in target_issues:
        item = by_id.get(str(issue.get("elementId")))
        if not item:
            continue
        problem = issue["problem"]
        metadata = item.setdefault("metadata", {})
        if problem in {"missingEditableText", "wrongOwnership"} and item.get("type") == "text":
            metadata.pop("suppressed", None)
            metadata.pop("suppressRender", None)
            metadata.pop("ownedBy", None)
            metadata.pop("sourceTextPreserved", None)
            metadata["reconstructionStrategy"] = "editable_text"
            touched_text.add(item["id"])
            erase_text.add(item["id"])
            changed_ids.add(item["id"])
            fallback_id = metadata.pop("fallbackAssetId", None)
            if fallback_id and fallback_id in by_id:
                by_id[fallback_id].setdefault("metadata", {})["suppressed"] = True
        elif problem in {"wrongBBox", "textOverlap"} and item.get("type") == "text":
            touched_text.add(item["id"])
            changed_ids.add(item["id"])
        elif problem == "duplicateText":
            other = by_id.get(str(issue.get("otherElementId")))
            if other and other.get("type") == "text":
                loser = item if float(item.get("confidence") or 0) < float(other.get("confidence") or 0) else other
                loser.setdefault("metadata", {})["suppressed"] = True
                changed_ids.add(loser["id"])
        elif problem == "ghosting" and item.get("type") == "text":
            touched_text.add(item["id"])
            erase_text.add(item["id"])
            changed_ids.add(item["id"])
        elif problem == "brokenChartOrModule" and item.get("type") == "image":
            bbox = [float(item.get("x") or 0), float(item.get("y") or 0), float(item.get("x") or 0) + float(item.get("width") or 0), float(item.get("y") or 0) + float(item.get("height") or 0)]
            target_issues.append({"elementId": item["id"], "problem": "assetBakedIntoBackground", "bbox": bbox})

    for text_id in touched_text:
        item = by_id[text_id]
        raw = (item.get("metadata") or {}).get("rawOCRBBox")
        if isinstance(raw, list) and len(raw) == 4:
            item["x"], item["y"] = float(raw[0]), float(raw[1])
    if touched_text:
        fit_text_to_ocr_lines({"slide": candidate["slide"], "elements": [by_id[text_id] for text_id in touched_text]})
    for issue in target_issues:
        if issue["problem"] != "textOverlap":
            continue
        item = by_id.get(str(issue.get("elementId")))
        raw = (item.get("metadata") or {}).get("rawOCRBBox") if item else None
        if isinstance(raw, list) and len(raw) == 4:
            item["width"] = max(4.0, float(raw[2]) - float(raw[0]))
            item["height"] = max(4.0, (float(raw[3]) - float(raw[1])) * 1.1)
    if erase_text:
        erase_editable_text_sources(candidate_bg, candidate, target_text_ids=erase_text, copy_asset_prefix=f"revision_{round_number}")

    baked = [issue for issue in target_issues if issue["problem"] == "assetBakedIntoBackground"][:4]
    if baked:
        modules = []
        width, height = float(candidate["slide"]["width"]), float(candidate["slide"]["height"])
        for issue in baked:
            x1, y1, x2, y2 = issue["bbox"]
            modules.append({"id": f"revision_{round_number}_{len(modules)}", "role": "complex_visual", "strategy": "whole_image", "bbox": {"left": x1 / width, "top": y1 / height, "width": (x2 - x1) / width, "height": (y2 - y1) / height}, "confidence": 1.0})
        scene = {"canvas": {"width": width, "height": height}, "vision": {"aiUsed": True, "reconstructionPlan": {"modules": modules}}, "elements": [
            {**copy.deepcopy(item), "bbox": {"left": item.get("x", 0), "top": item.get("y", 0), "width": item.get("width", 0), "height": item.get("height", 0)}}
            for item in candidate.get("elements", []) if item.get("type") != "background"
        ]}
        before_ids = {item["id"] for item in candidate.get("elements", [])}
        AIReconstructionPlanner().apply(scene, source, root / "assets", project_id, page, include_detected_visuals=False, asset_prefix=f"revision_{round_number}")
        candidate = ReconstructionPipeline._apply_refined_scene(None, candidate, scene)
        new_assets = [item for item in candidate["elements"] if item["id"] not in before_ids and item.get("type") == "image"]
        if new_assets:
            separate_foreground(candidate_bg, new_assets)
            changed_ids.update(item["id"] for item in new_assets)

    render_preview(candidate_bg, candidate, candidate_preview)
    score_after = run_visual_qa(source, candidate_preview, candidate_dir, candidate)
    detected_ids = {str(source_id) for item in baseline.get("elements", []) if item.get("type") == "text" for source_id in ((item.get("metadata") or {}).get("sourceOcrIds") or [item.get("id")]) if source_id}
    detected = max(int(score_before.get("detectedTextCount") or 0), len(detected_ids))
    coverage_before = float(score_before["editableTextCoverage"]) if "detectedTextCount" in score_before and "editableTextCoverage" in score_before else float(measure_text_coverage(baseline, detected)["editableTextCoverage"])
    coverage_after = float(measure_text_coverage(candidate, detected)["editableTextCoverage"])
    issues_after = collect_revision_issues(candidate, score_after, raw_scene)
    prior_regions = {item.get("elementId"): float(item.get("score") or 0) for item in score_before.get("regions", [])}
    improved = sorted({str(item.get("elementId")) for item in score_after.get("regions", []) if item.get("elementId") in changed_ids and float(item.get("score") or 0) > prior_regions.get(item.get("elementId"), 0) + 0.02})
    before_keys = {(item.get("problem"), item.get("elementId")) for item in issues_before}
    after_keys = {(item.get("problem"), item.get("elementId")) for item in issues_after}
    improved.extend(sorted({str(element_id) for _, element_id in before_keys - after_keys if element_id in changed_ids} - set(improved)))
    visual_delta = float(score_after.get("overall") or 0) - float(score_before.get("overall") or 0)
    coverage_delta = coverage_after - coverage_before
    visual_improved = visual_delta > 0.003 and coverage_delta >= -0.01
    editable_improved = coverage_delta > 0.02 and float(score_after.get("overall") or 0) >= 0.8 and visual_delta >= -0.12
    accepted = bool(changed_ids) and (visual_improved or editable_improved) and len(issues_after) <= len(issues_before) + 1
    report = {
        "revisionRound": round_number, "accepted": accepted,
        "targetedIssues": target_issues,
        "issuesBefore": issues_before, "issuesAfter": issues_after,
        "improvedRegions": improved, "visualBefore": float(score_before.get("overall") or 0),
        "visualAfter": float(score_after.get("overall") or 0),
        "editableCoverageBefore": coverage_before, "editableCoverageAfter": coverage_after,
    }
    history.append(report)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if accepted:
        shutil.copy2(candidate_bg, background)
        shutil.copy2(candidate_preview, preview)
        shutil.copy2(candidate_bg, root / ("clean_background.png" if page == 1 else f"clean_background_{page}.png"))
        shutil.copy2(candidate_preview, root / ("final_preview.png" if page == 1 else f"final_preview_{page}.png"))
        difference = candidate_dir / "difference.png"
        if difference.is_file():
            shutil.copy2(difference, root / "difference.png")
        needs_review = float(score_after.get("overall") or 0) < 0.85 or coverage_after < 0.8 or bool(issues_after)
        score_after.update({"revisionRound": round_number, "revisionStatus": "stagnated" if needs_review else "improved", "issuesBefore": issues_before, "issuesAfter": issues_after, "improvedRegions": improved, **measure_text_coverage(candidate, detected)})
        score_path.write_text(json.dumps(score_after, ensure_ascii=False, indent=2), encoding="utf-8")
        validation_path = root / ("visual_validation.json" if page == 1 else f"visual_validation_{page}.json")
        validation_path.write_text(json.dumps(score_after, ensure_ascii=False, indent=2), encoding="utf-8")
        problem_path = root / ("problem_report.json" if page == 1 else f"problem_report_{page}.json")
        problem_path.write_text(json.dumps({"revisionRound": round_number, "issuesAfter": issues_after, "editableTextCoverage": coverage_after, "overall": score_after.get("overall")}, ensure_ascii=False, indent=2), encoding="utf-8")
        store.save_slide(project_id, page, candidate)
    return {"layout": candidate if accepted else baseline, **report}


def collect_revision_issues(layout: dict, score: dict, scene: dict | None = None) -> list[dict]:
    issues: list[dict] = []
    for issue in score.get("issues", []):
        if issue.get("problem") == "textOverlap":
            issues.append({**issue, "problem": "textOverlap"})
        elif issue.get("problem") == "criticalRegionMismatch":
            issues.append({**issue, "problem": "brokenChartOrModule"})
    for item in layout.get("elements", []):
        meta = item.get("metadata") or {}
        if item.get("type") != "text" or not str(item.get("text") or "").strip() or item.get("role") in {"logo", "decorative_text"}:
            continue
        item_id = item["id"]
        if meta.get("duplicateSuppressed") or str(meta.get("ownedBy") or "").startswith("text_"):
            continue
        if any(meta.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            issues.append({"elementId": item_id, "problem": "missingEditableText"})
            continue
        raw = meta.get("rawOCRBBox")
        if isinstance(raw, list) and len(raw) == 4:
            tolerance = max(6, (float(raw[3]) - float(raw[1])) * 0.5)
            if abs(float(item.get("x") or 0) - float(raw[0])) > tolerance or abs(float(item.get("y") or 0) - float(raw[1])) > tolerance:
                issues.append({"elementId": item_id, "problem": "wrongBBox"})
        if meta.get("sourceTextPreserved") and not meta.get("sourceTextRecleaned"):
            issues.append({"elementId": item_id, "problem": "ghosting"})
    texts = [item for item in layout.get("elements", []) if item.get("type") == "text" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    for index, left in enumerate(texts):
        for right in texts[index + 1:]:
            if str(left.get("text") or "").strip().casefold() != str(right.get("text") or "").strip().casefold():
                continue
            left_box = (float(left.get("x") or 0), float(left.get("y") or 0), float(left.get("x") or 0) + float(left.get("width") or 0), float(left.get("y") or 0) + float(left.get("height") or 0))
            if _overlap_fraction(left_box, right) > 0.7:
                issues.append({"elementId": left["id"], "otherElementId": right["id"], "problem": "duplicateText"})
    if scene:
        active_images = [item for item in layout.get("elements", []) if item.get("type") == "image" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
        for region in scene.get("regions", []):
            if region.get("type") not in {"image", "figure", "chart", "table"} or float(region.get("confidence") or 0) < 0.6:
                continue
            box = region.get("bbox") or {}
            x, y, w, h = (float(box.get(key) or 0) for key in ("left", "top", "width", "height"))
            if w * h < 400:
                continue
            coverage = max((_overlap_fraction((x, y, x + w, y + h), item) for item in active_images), default=0)
            if coverage < 0.6:
                issues.append({"elementId": str(region.get("id")), "problem": "assetBakedIntoBackground", "bbox": [x, y, x + w, y + h]})
    return list({(item.get("problem"), item.get("elementId"), item.get("otherElementId")): item for item in issues}.values())


def _overlap_fraction(box: tuple[float, float, float, float], item: dict) -> float:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    right, bottom = x + float(item.get("width") or 0), y + float(item.get("height") or 0)
    overlap = max(0, min(box[2], right) - max(box[0], x)) * max(0, min(box[3], bottom) - max(box[1], y))
    return overlap / max(1, (box[2] - box[0]) * (box[3] - box[1]))
