"""Replay REAL saved OCR/Qwen/SAM evidence, verifying editable text and mattes.

No fake OCR, provider responses, keys or source-page special cases. New records
are diagnostic candidates; the old result is not overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "backend"))
from app.models.project_store import ProjectStore
from app.services.pptx.renderer import PPTXRenderer
from app.services.reconstruction.exclusive_ownership import (
    audit_raster_text,
    build_exclusive_scene,
)
from app.services.reconstruction.preservation_checks import (
    inspect_preservation,
)
from app.services.visual_qa.analyzer import render_preview, run_visual_qa


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("recorded_project",type=Path)
    parser.add_argument("--output-root",type=Path)
    args=parser.parse_args()
    previous=args.recorded_project.resolve()
    candidate=next((previous / "candidates").iterdir())
    store=ProjectStore(args.output_root or ROOT / "outputs")
    record=store.create("Real evidence - native text / coherent visual regression")
    root=store.root / record["id"]
    print(json.dumps({"projectId":record["id"],"output":str(root)}),flush=True)
    shutil.copytree(candidate / "assets",root / "assets")
    ocr=json.loads((candidate / "ocr_layout.json").read_text(encoding="utf-8"))
    scene=json.loads((candidate / "scene_raw.json").read_text(encoding="utf-8"))
    source=previous / "original.png"
    layout=build_exclusive_scene(source,ocr,scene,scene.get("segmentation",[]),root,record["id"],1)
    original=json.loads((previous / "slides/page_1.json").read_text(encoding="utf-8"))
    for key in ("ocrProvider","visionProvider","visionModel","segmentationProvider","conversionMode"):
        layout["metadata"][key]=original["metadata"].get(key)
    layout["metadata"]["recordedProviderSourceProject"]=previous.name
    local=copy.deepcopy(layout)
    for e in local["elements"]:
        if e.get("src"):
            e["src"]=str(root / ("backgrounds" if e["type"]=="background" else "assets") / e["src"].split("/")[-1])
    preview=root / "reconstructed_preview.png"
    render_preview(root / "backgrounds/page_1.png",local,preview)
    audit=audit_raster_text(source,layout,root)
    preservation=inspect_preservation(source,layout,root,preview)
    native=sum(e.get("owner")=="editable_text" for e in layout["elements"])
    images=sum(e.get("owner")=="movable_image" for e in layout["elements"])
    score=run_visual_qa(source,preview,root,local)
    score.update({"revisionRounds":0,"revisionRound":0,"revisionStatus":"automatic_revisions_paused",
                  "editableTextCoverage":native/max(1,len(layout["metadata"]["ownershipAudit"]["textDecisions"])),
                  "visionProvider":layout["metadata"].get("visionProvider"),"visionModel":layout["metadata"].get("visionModel"),
                  "preservationChecks":preservation,"nativeTextCount":native})
    (root / "visual_score.json").write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding="utf-8")
    report={"projectId":record["id"],"recordedRealProject":previous.name,"newApiCalls":0,"nativeTextCount":native,
            "movableImageCount":images,"audit":audit,"preservation":preservation,
            "assetPolicies":[e["metadata"].get("visualAssetPolicy") for e in layout["elements"] if e.get("owner")=="movable_image"]}
    (root / "native_recovery.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    assert native>0, "Raster-only reconstruction is not editable success"
    assert audit["missingAssetCount"]==audit["rasterNativeDuplicateTextCount"]==0
    assert preservation["unownedSourcePixels"]==preservation["multiplyOwnedSourcePixels"]==preservation["unexpectedVisualChangedPixels"]==0
    assert preservation["unownedGlyphPaperPixels"]==preservation["multiplyOwnedGlyphPaperPixels"]==0
    store.add_image(record["id"],{"path":str(source)})
    store.save_slide(record["id"],1,layout)
    shutil.copy2(source,root / "source.png")
    shutil.copy2(source,root / "original.png")
    pptx,_=PPTXRenderer().render_project(record["id"],[local],output_dir=root)
    deck=Presentation(pptx)
    assert sum(s.has_text_frame and bool(s.text) for s in deck.slides[0].shapes)==native
    # Export a genuine blank-text diagnostic: retained old glyphs will show here.
    blank=copy.deepcopy(local)
    blank["elements"]=[e for e in blank["elements"] if e.get("type")!="text"]
    blank_dir=root / "text_removed_test"
    PPTXRenderer().render_project(record["id"],[blank],output_dir=blank_dir)
    print(json.dumps({"projectId":record["id"],"nativeTextBoxesInPpt":native,"pptx":str(pptx)}),flush=True)


if __name__=="__main__":
    main()
