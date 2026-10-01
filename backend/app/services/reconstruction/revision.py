from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from app.config import INPAINT_PROVIDER, OUTPUTS_DIR
from app.services.inpainting.service import InpaintingService
from app.models.project_store import ProjectStore
from app.services.reconstruction.layered_background import separate_foreground
from app.services.reconstruction.asset_metrics import measure_movable_assets
from app.services.reconstruction.objectization_audit import audit_objectization, repair_missing_regions
from app.services.reconstruction.asset_ownership import is_badge_owned_text, restore_image_owned_text
from app.services.reconstruction.objectization_qa import repair_objectized_modules
from app.services.reconstruction.pipeline import ReconstructionPipeline
from app.services.reconstruction.planner import AIReconstructionPlanner
from app.services.reconstruction.revision_integrity import asset_path, assess_revision, inspect_assets, localize_project_assets, protected_visuals
from app.services.reconstruction.text_coverage import fit_text_to_ocr_lines, measure_text_coverage
from app.services.reconstruction.text_erasure import count_text_ghosting, erase_editable_text_sources
from app.services.visual_qa.analyzer import enrich_quality_score, render_preview, run_visual_qa

OBJECTIZATION_AUDIT_PROBLEMS = {"missingBackplate", "missingVisualObject", "blankVisualOwner", "visualContentMismatch", "assetBakedIntoBackground"}


def run_revision_loop(store: ProjectStore, project_id: str, page: int, max_rounds: int = 6) -> dict:
    """Continue local revisions while a high-quality page measurably improves."""
    root = store.root / project_id
    conversion_report = root / "conversion_report.json"
    mode = json.loads(conversion_report.read_text(encoding="utf-8")).get("mode") if conversion_report.is_file() else "standard"
    limit = max_rounds if mode in {"high_quality", "maximum"} else 1
    result: dict | None = None
    first: dict | None = None
    improved: set[str] = set()
    accepted_any = False
    consecutive_failures = 0
    for _ in range(limit):
        result = revise_problem_regions(store, project_id, page)
        first = first or result
        accepted_any = accepted_any or bool(result["accepted"])
        improved.update(result["improvedRegions"])
        consecutive_failures = 0 if result["accepted"] else consecutive_failures + 1
        if not result["issuesAfter"] or result.get("stagnationReason") == "no_targetable_issues" or consecutive_failures >= 2:
            break
    assert result is not None
    if consecutive_failures >= 2:
        result["stagnationReason"] = "consecutive_rounds_without_improvement"
        root = store.root / project_id
        for filename in ("visual_score.json" if page == 1 else f"visual_score_{page}.json", "visual_validation.json" if page == 1 else f"visual_validation_{page}.json", "problem_report.json" if page == 1 else f"problem_report_{page}.json"):
            path = root / filename
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["stagnationReason"] = result["stagnationReason"]
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif mode in {"high_quality", "maximum"} and consecutive_failures == 1 and result.get("stagnationReason") != "no_targetable_issues":
        result["stagnationReason"] = None
        root = store.root / project_id
        for filename in ("visual_score.json" if page == 1 else f"visual_score_{page}.json", "visual_validation.json" if page == 1 else f"visual_validation_{page}.json", "problem_report.json" if page == 1 else f"problem_report_{page}.json"):
            path = root / filename
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["stagnationReason"] = None
                if filename.startswith(("visual_score", "visual_validation")):
                    payload["revisionStatus"] = "improving"
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if accepted_any:
        result = {**result, "accepted": True, "layout": store.get_slide(project_id, page), "issuesBefore": first["issuesBefore"], "visualBefore": first["visualBefore"], "editableCoverageBefore": first["editableCoverageBefore"], "improvedRegions": sorted(improved)}
    return result


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
    localized = localize_project_assets(root, baseline)
    if localized:
        store.save_slide(project_id, page, baseline)
    baseline_assets = inspect_assets(root, baseline)
    if baseline_assets["missingAssetCount"]:
        raise ValueError("Current slide has missing image assets; repair the existing result before revision")
    asset_files_before = {path.resolve() for path in (root / "assets").glob("*") if path.is_file()}
    score_before = json.loads(score_path.read_text(encoding="utf-8"))
    history_path = root / f"revision_history_{page}.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    round_number = len(history) + 1
    scene_path = root / ("scene_raw.json" if page == 1 else f"scene_raw_{page}.json")
    raw_scene = json.loads(scene_path.read_text(encoding="utf-8")) if scene_path.is_file() else {}
    problem_path = root / ("problem_report.json" if page == 1 else f"problem_report_{page}.json")
    previous_report = json.loads(problem_path.read_text(encoding="utf-8")) if problem_path.is_file() else {}
    baseline_audit = audit_objectization(source, background, preview, baseline)
    score_before["objectizationAudit"] = baseline_audit
    saved_issues = previous_report.get("issuesAfter") if isinstance(previous_report.get("issuesAfter"), list) else []
    # Saved reports and scores describe the previous render. Pixel-derived
    # objectization issues must be revalidated against the current assets and
    # preview before a revision is allowed to create another visual object.
    carry_issues = [item for item in [*saved_issues, *collect_revision_issues(baseline, score_before, raw_scene)]
                    if item.get("problem") not in OBJECTIZATION_AUDIT_PROBLEMS]
    issues_before = list({(item.get("problem"), item.get("elementId")): item for item in [*carry_issues, *baseline_audit["issues"]]}.values())
    baseline_by_id = {str(item.get("id")): item for item in baseline.get("elements", [])}
    issues_before = [issue for issue in issues_before if not (
        issue.get("problem") == "missingEditableText"
        and is_badge_owned_text(baseline_by_id.get(str(issue.get("elementId")), {}), baseline)
    )]
    priority = {"ghosting": 0, "missingBackplate": 1, "missingVisualObject": 1, "blankVisualOwner": 1,
                "duplicateText": 2, "duplicateElement": 2, "wrongOwnership": 2,
                "visualContentMismatch": 3, "squareCutoutUnresolved": 3, "wrongZOrder": 3, "wrongBBox": 4,
                "textOverlap": 5, "missingEditableText": 6, "brokenChartOrModule": 7,
                "assetBakedIntoBackground": 8, "professionalInpaintingPending": 9, "backgroundResidual": 10}
    tried = {(item.get("problem"), item.get("elementId")) for attempt in history if not attempt.get("accepted") for item in attempt.get("targetedIssues", [])}
    ranked = sorted(issues_before, key=lambda item: priority.get(item["problem"], 9))
    target_issues = [item for item in ranked if (item.get("problem"), item.get("elementId")) not in tried][:4]
    candidate = copy.deepcopy(baseline)
    candidate_dir = root / "revisions" / f"round_{round_number}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_bg = candidate_dir / "background.png"
    candidate_preview = candidate_dir / "preview.png"
    shutil.copy2(background, candidate_bg)
    shutil.copy2(background, candidate_dir / "baseline_background.png")
    shutil.copy2(preview, candidate_dir / "baseline_preview.png")
    (candidate_dir / "baseline_layout.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    inpainting = InpaintingService(INPAINT_PROVIDER)
    local_cleaner = inpainting.clean_array if inpainting.provider.name == "local_lama" and any(issue["problem"] == "ghosting" for issue in target_issues) else None
    local_provider = inpainting.provider if inpainting.provider.name == "local_lama" else None
    touched_text: set[str] = set()
    erase_text: set[str] = set()
    changed_ids: set[str] = set()
    inpainted_regions = 0
    by_id = {str(item.get("id")): item for item in candidate.get("elements", [])}
    regional_analysis = _analyze_regions(source, preview, target_issues, by_id, candidate_dir)

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
        elif problem in {"duplicateText", "duplicateElement"}:
            other = by_id.get(str(issue.get("otherElementId")))
            if other and other.get("type") == item.get("type"):
                loser = item if float(item.get("confidence") or 0) < float(other.get("confidence") or 0) else other
                loser.setdefault("metadata", {})["suppressed"] = True
                changed_ids.add(loser["id"])
        elif problem == "ghosting" and item.get("type") == "text":
            touched_text.add(item["id"])
            erase_text.add(item["id"])
            changed_ids.add(item["id"])
        elif problem == "wrongZOrder" and item.get("type") == "text":
            owner = by_id.get(str(metadata.get("textCleanedFromAsset") or ""))
            if owner:
                item["zIndex"] = max(int(item.get("zIndex") or 0), int(owner.get("zIndex") or 0) + 1)
                changed_ids.add(item["id"])
        elif problem == "backgroundResidual" and item.get("type") == "image":
            changed_ids.add(item["id"])

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
        ghosting_ids = {str(issue.get("elementId")) for issue in target_issues if issue["problem"] == "ghosting"}
        erasure = erase_editable_text_sources(candidate_bg, candidate, complex_cleaner=local_cleaner, target_text_ids=erase_text, copy_asset_prefix=f"revision_{round_number}", project_root=root, protect_background_elements=True, force_asset_reclean_ids=ghosting_ids)
        inpainted_regions += erasure["backgroundTextErased"]

    residual_assets = [by_id[str(issue["elementId"])] for issue in target_issues if issue["problem"] == "backgroundResidual" and str(issue.get("elementId")) in by_id and not (by_id[str(issue["elementId"])].get("metadata") or {}).get("backgroundSeparated") and _background_contains_original(source, candidate_bg, by_id[str(issue["elementId"])])]
    if residual_assets:
        inpainted_regions += separate_foreground(candidate_bg, residual_assets, professional_provider=local_provider)
    baked = [issue for issue in target_issues if issue["problem"] in {"assetBakedIntoBackground", "brokenChartOrModule", "professionalInpaintingPending"} and isinstance(issue.get("bbox"), list) and not any(item.get("type") == "image" and _item_overlaps_box(item, issue["bbox"]) for item in protected_visuals(baseline))][:4]
    if baked:
        modules = []
        width, height = float(candidate["slide"]["width"]), float(candidate["slide"]["height"])
        for issue in baked:
            x1, y1, x2, y2 = issue["bbox"]
            modules.append({"id": f"revision_{round_number}_{len(modules)}", "role": "complex_visual", "strategy": "whole_image", "bbox": {"left": x1 / width, "top": y1 / height, "width": (x2 - x1) / width, "height": (y2 - y1) / height}, "confidence": 1.0})
        scene = {"canvas": {"width": width, "height": height}, "vision": {"aiUsed": True, "reconstructionPlan": {"modules": modules}}, "elements": [
            {**copy.deepcopy(item), "bbox": {"left": item.get("x", 0), "top": item.get("y", 0), "width": item.get("width", 0), "height": item.get("height", 0)}}
            for item in candidate.get("elements", []) if item.get("type") == "text" and any(_item_overlaps_box(item, issue["bbox"]) for issue in baked)
        ]}
        before_ids = {item["id"] for item in candidate.get("elements", [])}
        AIReconstructionPlanner().apply(scene, source, root / "assets", project_id, page, include_detected_visuals=False, asset_prefix=f"revision_{round_number}")
        candidate = ReconstructionPipeline._apply_refined_scene(None, candidate, scene)
        new_assets = [item for item in candidate["elements"] if item["id"] not in before_ids and item.get("type") == "image"]
        if new_assets:
            unseparated = [item for item in new_assets if _background_contains_original(source, candidate_bg, item)]
            if unseparated:
                inpainted_regions += separate_foreground(candidate_bg, unseparated, professional_provider=local_provider)
            changed_ids.update(item["id"] for item in new_assets)

    mismatch_boxes = [issue["bbox"] for issue in target_issues if issue.get("problem") == "visualContentMismatch" and isinstance(issue.get("bbox"), list)]
    restored_visuals = restore_image_owned_text(source, candidate, root / "assets", project_id,
                                                prefix=f"revision_{round_number}_owned_text", target_boxes=mismatch_boxes) if mismatch_boxes else []
    changed_ids.update(restored_visuals)
    restored_items = [by_id[item_id] for item_id in restored_visuals if item_id in by_id]
    repair_issues = [issue for issue in target_issues if not (
        issue.get("problem") == "visualContentMismatch" and isinstance(issue.get("bbox"), list)
        and any(_overlap_fraction(tuple(float(value) for value in issue["bbox"]), item) >= 0.85 for item in restored_items))]
    recovered = repair_missing_regions(source, candidate, repair_issues, root / "assets", project_id, round_number,
                                       asset_prefix=f"revision_page_{page}_round_{round_number}")
    changed_ids.update(str(item["id"]) for item in recovered)
    for issue in target_issues:
        if issue["problem"] != "visualContentMismatch" or not isinstance(issue.get("bbox"), list):
            continue
        x1, y1, x2, y2 = issue["bbox"]
        issue_area = max(1, (x2 - x1) * (y2 - y1))
        for old in baseline.get("elements", []):
            if old.get("id") in restored_visuals:
                continue
            if old.get("type") not in {"rectangle", "roundedRectangle", "ellipse", "image"}:
                continue
            ox1, oy1 = float(old.get("x") or 0), float(old.get("y") or 0)
            ox2, oy2 = ox1 + float(old.get("width") or 0), oy1 + float(old.get("height") or 0)
            old_area = max(1, (ox2 - ox1) * (oy2 - oy1))
            overlap = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
            if overlap / old_area >= 0.9 and old_area <= issue_area * 1.25:
                current = by_id.get(str(old.get("id")))
                if current is not None:
                    if old.get("type") == "image":
                        old_box = (ox1, oy1, ox2, oy2)
                        replacement = next((item for item in recovered
                                            if item.get("type") == "image"
                                            and (item.get("metadata") or {}).get("qaIssue") == "visualContentMismatch"
                                            and not (item.get("metadata") or {}).get("replacesAssetId")
                                            and _overlap_fraction(old_box, item) >= 0.9), None)
                        if replacement is None:
                            continue
                        current.setdefault("metadata", {})["replacedBy"] = replacement["id"]
                        replacement.setdefault("metadata", {})["replacesAssetId"] = str(old["id"])
                    current.setdefault("metadata", {}).update({"suppressed": True, "suppressRender": True})
                    changed_ids.add(str(current["id"]))
    for issue in target_issues:
        if issue["problem"] == "blankVisualOwner" and any(item["metadata"]["qaIssue"] == "blankVisualOwner" for item in recovered):
            old = by_id.get(str(issue.get("elementId")))
            if old:
                old.setdefault("metadata", {}).update({"suppressed": True, "suppressRender": True})
                changed_ids.add(str(old["id"]))
    square_ids = {str(issue.get("elementId")) for issue in target_issues if issue["problem"] == "squareCutoutUnresolved"}
    if square_ids:
        repaired = repair_objectized_modules(source, candidate, root / "assets", project_id, target_ids=square_ids)
        if repaired["repairedCutouts"]:
            changed_ids.update(square_ids)

    render_preview(candidate_bg, candidate, candidate_preview)
    target_boxes = [entry["bbox"] for entry in regional_analysis]
    for layout_version in (baseline, candidate):
        for item in layout_version.get("elements", []):
            if item.get("id") in changed_ids:
                x, y = float(item.get("x") or 0), float(item.get("y") or 0)
                target_boxes.append([x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)])
    integrity = assess_revision(root, baseline, candidate, background, candidate_bg, preview, candidate_preview, target_boxes)
    score_after = run_visual_qa(source, candidate_preview, candidate_dir, candidate)
    candidate_audit = audit_objectization(source, candidate_bg, candidate_preview, candidate, candidate_dir / "objectization_debug.png")
    if _visual_retention_regressed(baseline_audit, candidate_audit):
        integrity["integrityErrors"].append("source_visual_loss")
    ghosting_before = count_text_ghosting(source, background, baseline)
    ghosting_after = count_text_ghosting(source, candidate_bg, candidate)
    if ghosting_after > ghosting_before:
        integrity["integrityErrors"].append("text_ghosting_regressed")
    before_mismatch = sum(int(issue.get("pixelArea") or 0) for issue in baseline_audit["issues"] if issue.get("problem") == "visualContentMismatch")
    after_mismatch = sum(int(issue.get("pixelArea") or 0) for issue in candidate_audit["issues"] if issue.get("problem") == "visualContentMismatch")
    if any(issue.get("problem") == "visualContentMismatch" for issue in target_issues):
        if after_mismatch >= before_mismatch:
            integrity["integrityErrors"].append("visual_mismatch_not_improved")
    score_after["objectizationAudit"] = candidate_audit
    score_after.setdefault("issues", []).extend(candidate_audit["issues"])
    for key in ("visionProvider", "visionModel", "requestedVisionProvider", "ocrProvider", "conversionMode"):
        if key in score_before:
            score_after[key] = score_before[key]
    for key, field in (("localInpaintAttempts", "attempts"), ("localInpaintSuccesses", "successes"), ("localInpaintFallbacks", "failures")):
        score_after[key] = int(score_before.get(key) or 0) + int(getattr(inpainting.provider, field, 0))
    if local_provider:
        score_after["inpaintingProvider"] = "local_lama"
    detected_ids = {str(source_id) for item in baseline.get("elements", []) if item.get("type") == "text" for source_id in ((item.get("metadata") or {}).get("sourceOcrIds") or [item.get("id")]) if source_id}
    detected = max(int(score_before.get("detectedTextCount") or 0), len(detected_ids))
    coverage_before = float(score_before["editableTextCoverage"]) if "detectedTextCount" in score_before and "editableTextCoverage" in score_before else float(measure_text_coverage(baseline, detected)["editableTextCoverage"])
    coverage_after = float(measure_text_coverage(candidate, detected)["editableTextCoverage"])
    asset_metrics = measure_movable_assets(source, candidate_bg, candidate, (candidate.get("metadata") or {}).get("reconstructionPlan") or {})
    score_after.update(asset_metrics)
    enrich_quality_score(score_after, editable_coverage=coverage_after, movable_coverage=float(asset_metrics["movableVisualCoverage"]), ghosting_count=ghosting_after, background_residual_count=int(asset_metrics["backgroundResidualCount"]), professional_pending=int(score_before.get("professionalRepairPending") or 0))
    issues_after = collect_revision_issues(candidate, score_after, raw_scene)
    prior_regions = {item.get("elementId"): float(item.get("score") or 0) for item in score_before.get("regions", [])}
    improved = sorted({str(item.get("elementId")) for item in score_after.get("regions", []) if item.get("elementId") in changed_ids and float(item.get("score") or 0) > prior_regions.get(item.get("elementId"), 0) + 0.02})
    before_keys = {(item.get("problem"), item.get("elementId")) for item in issues_before}
    after_keys = {(item.get("problem"), item.get("elementId")) for item in issues_after}
    improved.extend(sorted({str(element_id) for _, element_id in before_keys - after_keys if element_id in changed_ids} - set(improved)))
    visual_delta = float(score_after.get("overall") or 0) - float(score_before.get("overall") or 0)
    coverage_delta = coverage_after - coverage_before
    visual_improved = visual_delta > 0.003 and coverage_delta >= -0.01
    editable_improved = coverage_delta > 0.02 and visual_delta >= -0.12
    critical = {"missingEditableText", "ghosting", "duplicateText", "wrongOwnership", "missingBackplate", "missingVisualObject", "blankVisualOwner", "visualContentMismatch", "assetBakedIntoBackground", "backgroundResidual", "brokenChartOrModule", "wrongZOrder"}
    resolved_critical = any(problem in critical for problem, _ in before_keys - after_keys)
    local_improved = resolved_critical and visual_delta >= -0.005 and coverage_delta >= -0.01
    restored_mismatch = bool(restored_visuals) and before_mismatch > 0 and after_mismatch < before_mismatch * 0.2 and coverage_delta >= -0.01
    accepted = not integrity["integrityErrors"] and bool(changed_ids) and (visual_improved or editable_improved or local_improved or restored_mismatch) and len(issues_after) <= len(issues_before) + 1
    reported_integrity = {**integrity, "candidateMissingAssetCount": integrity["missingAssetCount"], "candidateAssetsAfter": integrity["assetsAfter"],
                          "missingAssetCount": integrity["missingAssetCount"] if accepted else baseline_assets["missingAssetCount"],
                          "assetsAfter": integrity["assetsAfter"] if accepted else baseline_assets["assets"],
                          "preservedAssetCount": integrity["preservedAssetCount"] if accepted else baseline_assets["assets"],
                          "replacedAssetCount": integrity["replacedAssetCount"] if accepted else 0,
                          "backgroundWhiteAfter": integrity["backgroundWhiteAfter"] if accepted else integrity["backgroundWhiteBefore"],
                          "previewWhiteAfter": integrity["previewWhiteAfter"] if accepted else integrity["previewWhiteBefore"]}
    stagnation_reason = None if accepted else "integrity_check_failed" if integrity["integrityErrors"] else "no_targetable_issues" if not target_issues else "no_supported_change" if not changed_ids else "no_measurable_improvement"
    report = {
        "revisionRound": round_number, "accepted": accepted,
        "localizedAssetCount": localized,
        "targetedIssues": target_issues,
        "issuesBefore": issues_before, "issuesAfter": issues_after if accepted else issues_before,
        "improvedRegions": improved if accepted else [], "visualBefore": float(score_before.get("overall") or 0),
        "visualAfter": float(score_after.get("overall") or 0),
        "editableCoverageBefore": coverage_before, "editableCoverageAfter": coverage_after,
        "regionalAnalysis": regional_analysis, "candidateIssuesAfter": issues_after, "stagnationReason": stagnation_reason,
        **reported_integrity, "inpaintedRegions": inpainted_regions if accepted else 0, "attemptedInpaintedRegions": inpainted_regions, "rollbackTriggered": not accepted,
        "discardedCandidateAssets": 0,
    }
    if accepted:
        needs_review = float(score_after.get("overall") or 0) < 0.85 or coverage_after < 0.95 or bool(issues_after)
        score_after.update({"revisionRound": round_number, "revisionStatus": "improving" if needs_review else "improved", "issuesBefore": issues_before, "issuesAfter": issues_after, "improvedRegions": improved, "stagnationReason": None, **measure_text_coverage(candidate, detected), **reported_integrity, "inpaintedRegions": inpainted_regions, "rollbackTriggered": False})
        validation_path = root / ("visual_validation.json" if page == 1 else f"visual_validation_{page}.json")
        accepted_problem = {"revisionRound": round_number, "issuesBefore": issues_before, "issuesAfter": issues_after, "improvedRegions": improved, "stagnationReason": None, "editableTextCoverage": coverage_after, "overall": score_after.get("overall")}
        replacements = {background: candidate_bg, preview: candidate_preview,
                        root / ("clean_background.png" if page == 1 else f"clean_background_{page}.png"): candidate_bg,
                        root / ("final_preview.png" if page == 1 else f"final_preview_{page}.png"): candidate_preview}
        difference = candidate_dir / "difference.png"
        if difference.is_file():
            replacements[root / "difference.png"] = difference
        _commit_revision(store, project_id, page, candidate, candidate_dir, replacements,
                         {score_path: score_after, validation_path: score_after, problem_path: accepted_problem})
    else:
        report["discardedCandidateAssets"] = _discard_candidate_assets(root, candidate, asset_files_before)
        score_before.update({"revisionRound": round_number, "revisionStatus": "stagnated", "issuesBefore": issues_before, "issuesAfter": issues_before, "improvedRegions": [], "stagnationReason": stagnation_reason, **reported_integrity, "inpaintedRegions": 0, "attemptedInpaintedRegions": inpainted_regions, "rollbackTriggered": True})
        score_path.write_text(json.dumps(score_before, ensure_ascii=False, indent=2), encoding="utf-8")
        validation_path = root / ("visual_validation.json" if page == 1 else f"visual_validation_{page}.json")
        validation_path.write_text(json.dumps(score_before, ensure_ascii=False, indent=2), encoding="utf-8")
        problem_path.write_text(json.dumps({"revisionRound": round_number, "issuesBefore": issues_before, "issuesAfter": issues_before, "improvedRegions": [], "stagnationReason": stagnation_reason, "editableTextCoverage": coverage_before, "overall": score_before.get("overall")}, ensure_ascii=False, indent=2), encoding="utf-8")
    history.append(report)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (candidate_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"layout": candidate if accepted else baseline, **report}


def _discard_candidate_assets(root: Path, candidate: dict, preexisting: set[Path]) -> int:
    """Remove only new files referenced by a rejected candidate."""
    asset_dir = (root / "assets").resolve()
    discard: set[Path] = set()
    for item in candidate.get("elements", []):
        if item.get("type") != "image":
            continue
        path = asset_path(root, item.get("src"))
        if path is not None:
            resolved = path.resolve()
            if resolved.parent == asset_dir and resolved not in preexisting:
                discard.add(resolved)
    for path in discard:
        path.unlink(missing_ok=True)
    return len(discard)


def _visual_retention_regressed(before: dict, after: dict) -> bool:
    salient = max(int(before.get("salientVisualPixels") or 0), int(after.get("salientVisualPixels") or 0))
    # Page-wide coverage can round to 1.0 even when a small but meaningful
    # icon disappears. Keep an absolute pixel guard alongside the ratio.
    missing_growth = int(after.get("missingVisualPixels") or 0) - int(before.get("missingVisualPixels") or 0)
    missing_threshold = max(24, min(96, round(salient * 0.0002)))
    largest_after = int(after.get("largestMissingVisualRegion", missing_growth))
    largest_before = int(before.get("largestMissingVisualRegion", 0))
    lost_to_white = (missing_growth > missing_threshold and largest_after > missing_threshold
                     and largest_after > largest_before + 24)
    mismatch_growth = int(after.get("visualMismatchPixels") or 0) - int(before.get("visualMismatchPixels") or 0)
    lost_to_wrong_color = salient > 0 and mismatch_growth > max(24, round(salient * 0.005))
    pale_before = int(before.get("paleAssetGapPixels") or 0)
    pale_growth = int(after.get("paleAssetGapPixels") or 0) - pale_before
    lost_pale_support = pale_growth > max(48, min(300, round(pale_before * 0.015)))
    return lost_to_white or lost_to_wrong_color or lost_pale_support


def _commit_revision(store: ProjectStore, project_id: str, page: int, candidate: dict, candidate_dir: Path, replacements: dict[Path, Path], payloads: dict[Path, dict]) -> None:
    backups: dict[Path, Path | None] = {}
    targets = [*replacements, *payloads]
    for index, target in enumerate(targets):
        backup = candidate_dir / f"commit_backup_{index}{target.suffix}"
        if target.is_file():
            shutil.copy2(target, backup)
            backups[target] = backup
        else:
            backups[target] = None
    try:
        for target, source in replacements.items():
            shutil.copy2(source, target)
        for target, payload in payloads.items():
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        store.save_slide(project_id, page, candidate)
    except Exception:
        for target, backup in backups.items():
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                shutil.copy2(backup, target)
        raise


def collect_revision_issues(layout: dict, score: dict, scene: dict | None = None) -> list[dict]:
    issues: list[dict] = []
    for issue in score.get("issues", []):
        if issue.get("problem") in {"textOverlap", "wrongBBox", "wrongZOrder", "duplicateText", "duplicateElement", "imageDistortion", "moduleBoundary", "brokenChartOrModule", "professionalInpaintingPending", "missingBackplate", "missingVisualObject", "blankVisualOwner", "visualContentMismatch", "squareCutoutUnresolved"}:
            issues.append(issue)
        elif issue.get("problem") == "criticalRegionMismatch":
            item = next((element for element in layout.get("elements", []) if element.get("id") == issue.get("elementId")), None)
            if item and item.get("type") in {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"}:
                x, y = float(item.get("x") or 0), float(item.get("y") or 0)
                issues.append({**issue, "problem": "brokenChartOrModule", "bbox": [x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)]})
    for item in layout.get("elements", []):
        meta = item.get("metadata") or {}
        if item.get("type") != "text" or not str(item.get("text") or "").strip() or item.get("role") in {"logo", "decorative_text"}:
            continue
        item_id = item["id"]
        if is_badge_owned_text(item, layout):
            continue
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
        if meta.get("ghostingDetected") or meta.get("sourceTextPreserved") and not meta.get("sourceTextRecleaned"):
            issues.append({"elementId": item_id, "problem": "ghosting"})
        owner_id = meta.get("textCleanedFromAsset")
        if owner_id:
            owner = next((element for element in layout.get("elements", []) if element.get("id") == owner_id), None)
            if owner and int(item.get("zIndex") or 0) <= int(owner.get("zIndex") or 0):
                issues.append({"elementId": item_id, "problem": "wrongZOrder"})
            if owner and not (owner.get("metadata") or {}).get("textCleaned"):
                issues.append({"elementId": item_id, "problem": "wrongOwnership"})
    texts = [item for item in layout.get("elements", []) if item.get("type") == "text" and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    for index, left in enumerate(texts):
        for right in texts[index + 1:]:
            if str(left.get("text") or "").strip().casefold() != str(right.get("text") or "").strip().casefold():
                continue
            left_box = (float(left.get("x") or 0), float(left.get("y") or 0), float(left.get("x") or 0) + float(left.get("width") or 0), float(left.get("y") or 0) + float(left.get("height") or 0))
            if _overlap_fraction(left_box, right) > 0.7:
                issues.append({"elementId": left["id"], "otherElementId": right["id"], "problem": "duplicateText"})
    active_visuals = [item for item in layout.get("elements", []) if item.get("type") in {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"} and not any((item.get("metadata") or {}).get(key) for key in ("suppressed", "suppressRender", "ownedBy"))]
    for index, left in enumerate(active_visuals):
        left_box = (float(left.get("x") or 0), float(left.get("y") or 0), float(left.get("x") or 0) + float(left.get("width") or 0), float(left.get("y") or 0) + float(left.get("height") or 0))
        for right in active_visuals[index + 1:]:
            if left.get("type") == right.get("type") and left.get("src") == right.get("src") and (left.get("style") or {}) == (right.get("style") or {}) and _overlap_fraction(left_box, right) >= 0.9:
                issues.append({"elementId": left["id"], "otherElementId": right["id"], "problem": "duplicateElement"})
    if int(score.get("backgroundResidualCount") or 0) > 0:
        for asset in active_visuals:
            if asset.get("type") == "image" and not (asset.get("metadata") or {}).get("backgroundSeparated"):
                issues.append({"elementId": asset["id"], "problem": "backgroundResidual"})
    audit = score.get("objectizationAudit") or {}
    background_is_clear = bool(audit.get("whiteBackground")) and int(audit.get("backgroundResidualRegions") or 0) == 0
    if scene and not background_is_clear:
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


def _item_overlaps_box(item: dict, box: list[float]) -> bool:
    x, y = float(item.get("x") or 0), float(item.get("y") or 0)
    w, h = float(item.get("width") or 0), float(item.get("height") or 0)
    overlap = max(0.0, min(x + w, float(box[2])) - max(x, float(box[0]))) * max(0.0, min(y + h, float(box[3])) - max(y, float(box[1])))
    return overlap / max(1.0, min(w * h, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))) >= 0.15


def _background_contains_original(source_path: Path, background_path: Path, item: dict) -> bool:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if source is None or background is None or source.shape != background.shape:
        return False
    height, width = source.shape[:2]
    x1, y1 = max(0, int(float(item.get("x") or 0))), max(0, int(float(item.get("y") or 0)))
    x2 = min(width, x1 + int(float(item.get("width") or 0)))
    y2 = min(height, y1 + int(float(item.get("height") or 0)))
    if x2 <= x1 or y2 <= y1:
        return False
    original = source[y1:y2, x1:x2]
    current = background[y1:y2, x1:x2]
    return float(np.mean(cv2.absdiff(original, current))) < 35 and float(np.std(original)) > 10


def _analyze_regions(source_path: Path, preview_path: Path, issues: list[dict], by_id: dict[str, dict], output: Path) -> list[dict]:
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    previous = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
    if source is None or previous is None:
        return []
    height, width = source.shape[:2]
    analysis = []
    for index, issue in enumerate(issues):
        box = issue.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            item = by_id.get(str(issue.get("elementId")))
            if not item:
                continue
            x, y = float(item.get("x") or 0), float(item.get("y") or 0)
            box = [x, y, x + float(item.get("width") or 0), y + float(item.get("height") or 0)]
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        original_crop = source[y1:y2, x1:x2]
        previous_crop = previous[y1:y2, x1:x2]
        original_edges = cv2.Canny(original_crop, 60, 160)
        previous_edges = cv2.Canny(previous_crop, 60, 160)
        pixel_difference = float(np.mean(cv2.absdiff(original_crop, previous_crop))) / 255
        edge_difference = float(np.mean(cv2.absdiff(original_edges, previous_edges))) / 255
        cv2.imwrite(str(output / f"region_{index + 1}_source.png"), original_crop)
        cv2.imwrite(str(output / f"region_{index + 1}_previous.png"), previous_crop)
        analysis.append({"elementId": issue.get("elementId"), "problem": issue.get("problem"), "bbox": [x1, y1, x2, y2], "pixelDifference": round(pixel_difference, 4), "edgeDifference": round(edge_difference, 4), "strategyChange": "movable_image_with_editable_text" if issue.get("problem") in {"brokenChartOrModule", "assetBakedIntoBackground"} else "editable_text_repair" if issue.get("problem") in {"missingEditableText", "ghosting", "wrongBBox", "textOverlap"} else "ownership_repair"})
    return analysis
