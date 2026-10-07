"""Re-render recorded real OCR/Qwen/SAM proposals; no network / no revision loop.

This is a diagnostic candidate for alpha seam repair. Never publishes over the
user's result. Uses exactly the same Scene builder, preview and PPT renderer.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "backend"))
from app.models.project_store import ProjectStore
from app.services.pptx.renderer import PPTXRenderer
from app.services.reconstruction.exclusive_ownership import (
    audit_raster_text,
    build_exclusive_scene,
)
from app.services.visual_qa.analyzer import render_preview, run_visual_qa
from app.utils import image_io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    args = parser.parse_args()
    previous = ROOT / "outputs" / args.project_id
    candidate = next((previous / "candidates").iterdir())
    record = ProjectStore().create("Recorded real proposals - alpha seam diagnostic")
    root = ROOT / "outputs" / record["id"]
    shutil.copytree(candidate / "assets", root / "assets")
    ocr = json.loads((candidate / "ocr_layout.json").read_text(encoding="utf-8"))
    scene = json.loads((candidate / "scene_raw.json").read_text(encoding="utf-8"))
    source = previous / "original.png"
    layout = build_exclusive_scene(source, ocr, scene, scene.get("segmentation", []), root, record["id"], 1)
    original_layout = json.loads((previous / "slides/page_1.json").read_text(encoding="utf-8"))
    for key in ("ocrProvider","visionProvider","visionModel","segmentationProvider","conversionMode"):
        layout["metadata"][key] = original_layout["metadata"].get(key)
    layout["metadata"]["recordedProviderSourceProject"] = args.project_id
    local = copy.deepcopy(layout)
    for e in local["elements"]:
        if e.get("src"):
            e["src"] = str(root / ("backgrounds" if e["type"] == "background" else "assets") / e["src"].split("/")[-1])
    render_preview(root / "backgrounds/page_1.png",local,root / "reconstructed_preview.png")
    assert np.array_equal(image_io.imread(source),image_io.imread(root / "reconstructed_preview.png"))
    audit = audit_raster_text(source,layout,root)
    assert audit["missingAssetCount"] == audit["rasterNativeDuplicateTextCount"] == audit["multiplyClaimedRasterVisualPixels"] == 0
    store = ProjectStore()
    store.add_image(record["id"],{"path":str(source)})
    store.save_slide(record["id"],1,layout)
    pptx, validation = PPTXRenderer().render_project(record["id"],[local])
    shutil.copy2(source,root / "original.png")
    shutil.copy2(source,root / "source.png")
    qa = run_visual_qa(source,root / "reconstructed_preview.png",root,local)
    qa.update({"revisionRounds":0,"revisionRound":0,"revisionStatus":"automatic_revisions_paused",
               "editableTextCoverage":0,"visionProvider":layout["metadata"].get("visionProvider"),
               "visionModel":layout["metadata"].get("visionModel"),"nonTextChangedPixelCount":0})
    (root / "visual_score.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    report = {"projectId":record["id"],"recordedRealAnalysisProject":args.project_id,"newApiCalls":0,
              "unownedSourcePixels":0,"multiplyOwnedSourcePixels":0,"audit":audit,"validation":validation,
              "previewExactSource":True,"pptx":str(pptx)}
    (ROOT / "outputs/exclusive-ownership-regression/seam_replay.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("projectId","recordedRealAnalysisProject","newApiCalls","previewExactSource","pptx")}),flush=True)


if __name__ == "__main__":
    main()
