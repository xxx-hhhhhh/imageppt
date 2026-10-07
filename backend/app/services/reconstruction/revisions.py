from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path

from app.services.fusion.adjustment_validator import apply_safe_adjustments
from app.services.scene.ownership import canonicalize, resolve_asset, scene_view
from app.services.visual_qa.analyzer import render_preview, run_visual_qa
from app.services.visual_qa.preservation import preservation_qa, revision_gate


class RevisionManager:
    def __init__(self, output_root: Path, project_id: str, page: int) -> None:
        self.output_root = output_root
        self.root = output_root / project_id / "revisions" / f"page_{page}"
        self.root.mkdir(parents=True, exist_ok=True)

    def snapshot(self, layout: dict, source: Path, background: Path, status: str, reasons: list[str] | None = None) -> tuple[Path, dict]:
        # Unique directories never overwrite an accepted or rejected revision.
        revision = self.root / f"{len(list(self.root.glob('r_*'))):04d}_{uuid.uuid4().hex[:8]}"
        revision = revision.with_name("r_" + revision.name)
        revision.mkdir()
        canonicalize(layout)
        (revision / "scene.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copy2(background, revision / "background.png")
        assets = revision / "assets"
        assets.mkdir()
        for item in layout.get("elements", []):
            for key in ("src", "mask"):
                path = resolve_asset(item.get(key), self.output_root)
                if path and path.is_file():
                    shutil.copy2(path, assets / path.name)
        render_preview(background, layout, revision / "preview.png", output_root=self.output_root)
        qa = run_visual_qa(source, revision / "preview.png", revision, layout)
        qa.update(preservation_qa(source, revision / "preview.png", background, layout, self.output_root))
        (revision / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
        self.decision(revision, status, reasons or [])
        return revision, qa

    def decision(self, revision: Path, status: str, reasons: list[str]) -> None:
        (revision / "decision.json").write_text(json.dumps({"status": status, "reasons": reasons}, ensure_ascii=False, indent=2), encoding="utf-8")

    def propose(self, layout: dict, critic: dict) -> dict:
        scene = scene_view(copy.deepcopy(layout))
        scene = apply_safe_adjustments(scene, critic)
        by_id = {item["id"]: item for item in scene["elements"]}
        candidate = copy.deepcopy(layout)
        for item in candidate["elements"]:
            updated = by_id[item["id"]]
            box = updated["bbox"]
            for key, box_key in (("x", "left"), ("y", "top"), ("width", "width"), ("height", "height")):
                item[key] = box[box_key]
            item["style"] = copy.deepcopy(updated.get("style") or {})
        candidate.setdefault("metadata", {})["criticAdjustments"] = scene.get("criticAdjustments", {})
        return canonicalize(candidate)

    def revise(self, layout: dict, source: Path, background: Path, critic: dict, previous_qa: dict) -> tuple[dict, dict, dict]:
        candidate = self.propose(layout, critic)
        revision, qa = self.snapshot(candidate, source, background, "candidate")
        accepted, reasons = revision_gate(previous_qa, qa, layout, candidate)
        self.decision(revision, "committed" if accepted else "rolled_back", reasons)
        result = candidate if accepted else layout
        result.setdefault("metadata", {})["lastRevision"] = {"id": revision.name, "status": "committed" if accepted else "rolled_back", "reasons": reasons}
        return result, qa if accepted else previous_qa, {"id": revision.name, "accepted": accepted, "reasons": reasons}
