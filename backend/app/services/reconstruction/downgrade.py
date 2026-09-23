from __future__ import annotations

import json
from pathlib import Path

from app.config import OUTPUTS_DIR
from app.models.project_store import ProjectStore
from app.services.reconstruction.pipeline import ReconstructionPipeline
from app.services.reconstruction.planner import AIReconstructionPlanner
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


def downgrade_problem_regions(store: ProjectStore, project_id: str, page: int) -> dict:
    """Honor the user's fallback choice for visual regions flagged by QA."""
    output = OUTPUTS_DIR / project_id
    source = output / ("source.png" if page == 1 else f"source_{page}.png")
    background = output / "backgrounds" / f"page_{page}.png"
    validation = output / ("visual_validation.json" if page == 1 else f"visual_validation_{page}.json")
    if not source.is_file() or not validation.is_file():
        raise FileNotFoundError("Page artifacts are not available")
    layout = store.get_slide(project_id, page)
    report = json.loads(validation.read_text(encoding="utf-8"))
    width, height = int(layout["slide"]["width"]), int(layout["slide"]["height"])
    by_id = {item["id"]: item for item in layout.get("elements", [])}
    modules = []
    for issue in report.get("issues", []):
        if issue.get("problem") != "criticalRegionMismatch":
            continue
        item = by_id.get(issue.get("elementId"))
        if not item or item.get("type") not in {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"}:
            continue
        if (item.get("metadata") or {}).get("suppressed"):
            continue
        x, y, w, h = (float(item.get(key, 0)) for key in ("x", "y", "width", "height"))
        if w < 24 or h < 24:
            continue
        modules.append({"id": f"fallback_{page}_{len(modules) + 1}", "role": item.get("role") or "complex_visual", "strategy": "whole_image", "bbox": {"left": max(0, x / width), "top": max(0, y / height), "width": min(1, w / width), "height": min(1, h / height)}, "confidence": 1.0})
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
    preview = output / ("reconstructed_preview.png" if page == 1 else f"reconstructed_preview_{page}.png")
    render_preview(background, revised, preview)
    score = run_visual_qa(source, preview, output, revised)
    score["revisionStatus"] = "user_downgraded"
    validation.write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    store.save_slide(project_id, page, revised)
    return revised
