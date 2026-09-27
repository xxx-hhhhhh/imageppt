from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.models.project_store import ProjectStore
from app.services.reconstruction.pipeline import ReconstructionPipeline
from app.services.reconstruction.planner import AIReconstructionPlanner
from app.services.reconstruction.layered_background import separate_foreground
from app.services.reconstruction.text_erasure import erase_editable_text_sources
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


def downgrade_problem_regions(store: ProjectStore, project_id: str, page: int) -> dict:
    """Honor the user's fallback choice for visual regions flagged by QA."""
    output = store.root / project_id
    source = output / ("source.png" if page == 1 else f"source_{page}.png")
    background = output / "backgrounds" / f"page_{page}.png"
    validation = output / ("visual_validation.json" if page == 1 else f"visual_validation_{page}.json")
    if not source.is_file() or not validation.is_file():
        raise FileNotFoundError("Page artifacts are not available")
    layout = store.get_slide(project_id, page)
    report = json.loads(validation.read_text(encoding="utf-8"))
    problem_path = output / ("problem_report.json" if page == 1 else f"problem_report_{page}.json")
    saved_problems = json.loads(problem_path.read_text(encoding="utf-8")) if problem_path.is_file() else {}
    width, height = int(layout["slide"]["width"]), int(layout["slide"]["height"])
    by_id = {item["id"]: item for item in layout.get("elements", [])}
    modules = []
    for issue in (saved_problems.get("issuesAfter") or report.get("issues", [])):
        if issue.get("problem") not in {"criticalRegionMismatch", "brokenChartOrModule", "assetBakedIntoBackground", "backgroundResidual"}:
            continue
        item = by_id.get(issue.get("elementId"))
        if item and (item.get("type") not in {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"} or (item.get("metadata") or {}).get("suppressed")):
            continue
        if item:
            x, y, w, h = (float(item.get(key, 0)) for key in ("x", "y", "width", "height"))
        elif isinstance(issue.get("bbox"), list) and len(issue["bbox"]) == 4:
            x, y, right, bottom = map(float, issue["bbox"])
            w, h = right - x, bottom - y
        else:
            continue
        if w < 24 or h < 24:
            continue
        modules.append({"id": f"fallback_{page}_{len(modules) + 1}", "role": (item or {}).get("role") or "complex_visual", "strategy": "whole_image", "bbox": {"left": max(0, x / width), "top": max(0, y / height), "width": min(1, w / width), "height": min(1, h / height)}, "confidence": 1.0})
        if len(modules) >= 8:
            break
    if not modules:
        raise ValueError("No visual problem region is available for downgrade")
    scene = {"canvas": {"width": width, "height": height}, "vision": {"aiUsed": True, "reconstructionPlan": {"modules": modules}}, "elements": []}
    for item in layout.get("elements", []):
        scene["elements"].append({**item, "bbox": {"left": item.get("x", 0), "top": item.get("y", 0), "width": item.get("width", 0), "height": item.get("height", 0)}})
    stats = AIReconstructionPlanner().apply(scene, source, output / "assets", project_id, page)
    if not stats["wholeImageRegions"]:
        raise ValueError("Problem regions could not be converted")
    revised = ReconstructionPipeline._apply_refined_scene(None, layout, scene)
    new_assets = [item for item in revised.get("elements", []) if item.get("id") not in by_id and item.get("type") == "image"]
    candidate_background = output / "revisions" / f"fallback_{page}" / "background.png"
    candidate_background.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(background, candidate_background)
    separate_foreground(candidate_background, new_assets)
    erase_editable_text_sources(candidate_background, revised)
    preview = output / ("reconstructed_preview.png" if page == 1 else f"reconstructed_preview_{page}.png")
    candidate_preview = candidate_background.with_name("preview.png")
    render_preview(candidate_background, revised, candidate_preview)
    shutil.copy2(candidate_background, background)
    shutil.copy2(candidate_preview, preview)
    score = run_visual_qa(source, preview, output, revised)
    score["revisionStatus"] = "user_downgraded"
    score["stagnationReason"] = None
    validation.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / ("visual_score.json" if page == 1 else f"visual_score_{page}.json")).write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    if problem_path.is_file():
        saved_problems["stagnationReason"] = None
        saved_problems["revisionStatus"] = "user_downgraded"
        problem_path.write_text(json.dumps(saved_problems, ensure_ascii=False, indent=2), encoding="utf-8")
    store.save_slide(project_id, page, revised)
    return revised
