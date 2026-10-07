"""Repair slide URLs produced by the old revision asset-root bug.

Run from backend/: python scripts/repair_revision_assets.py PROJECT_ID PAGE
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import OUTPUTS_DIR
from app.models.project_store import ProjectStore
from app.services.reconstruction.revision_integrity import inspect_assets, recover_legacy_revision_assets
from app.services.visual_qa.analyzer import render_preview


def repair(project_id: str, page: int) -> dict:
    store = ProjectStore(OUTPUTS_DIR)
    root = OUTPUTS_DIR / project_id
    baseline = store.get_slide(project_id, page)
    candidate = copy.deepcopy(baseline)
    recovered = recover_legacy_revision_assets(root, candidate)
    if not recovered:
        return {"recovered": 0, **inspect_assets(root, baseline)}
    assets = inspect_assets(root, candidate)
    if assets["missingAssetCount"]:
        raise ValueError(f"Recovery left {assets['missingAssetCount']} missing assets; slide was not changed")
    workspace = root / "revisions" / f"recovery_{uuid4().hex[:8]}"
    workspace.mkdir(parents=True)
    background = root / "backgrounds" / f"page_{page}.png"
    preview = root / ("reconstructed_preview.png" if page == 1 else f"reconstructed_preview_{page}.png")
    final = root / ("final_preview.png" if page == 1 else f"final_preview_{page}.png")
    candidate_preview = workspace / "preview.png"
    render_preview(background, candidate, candidate_preview)
    (workspace / "baseline_layout.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    backups = {}
    for target in (preview, final):
        backup = workspace / f"baseline_{target.name}"
        if target.is_file():
            shutil.copy2(target, backup)
            backups[target] = backup
    try:
        shutil.copy2(candidate_preview, preview)
        shutil.copy2(candidate_preview, final)
        store.save_slide(project_id, page, candidate)
    except Exception:
        for target, backup in backups.items():
            shutil.copy2(backup, target)
        store.save_slide(project_id, page, baseline)
        raise
    report = {"recovered": recovered, **assets, "projectId": project_id, "page": page}
    (workspace / "recovery_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parser.add_argument("page", type=int)
    args = parser.parse_args()
    print(json.dumps(repair(args.project_id, args.page), ensure_ascii=False))
