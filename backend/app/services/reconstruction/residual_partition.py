"""Split sparse, oversized residual crops without changing their rendered pixels."""

from __future__ import annotations

import copy
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def partition_sparse_residuals(layout: dict, asset_dir: Path, project_id: str) -> dict[str, int]:
    slide = layout.get("slide") or {}
    page_area = float(slide.get("width") or 0) * float(slide.get("height") or 0)
    stats = {"partitionedResidualAssets": 0, "residualPartsCreated": 0}
    if page_area <= 0:
        return stats
    elements = layout.get("elements") or []
    for item in list(elements):
        metadata = item.setdefault("metadata", {})
        if (item.get("type") != "image" or metadata.get("layerRole") != "residual"
                or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy"))):
            continue
        area = float(item.get("width") or 0) * float(item.get("height") or 0)
        if area < page_area * 0.15:
            continue
        original_path = asset_dir / Path(str(item.get("src") or "")).name
        image = cv2.imread(str(original_path), cv2.IMREAD_UNCHANGED) if original_path.is_file() else None
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            continue
        height, width = image.shape[:2]
        visible = image[:, :, 3] > 32
        visible_count = int(np.count_nonzero(visible))
        if visible_count < 2000 or visible_count / max(1, width * height) >= 0.45:
            continue
        count, labels, components, _ = cv2.connectedComponentsWithStats(np.uint8(visible), 8)
        frame = any(
            int(components[index, cv2.CC_STAT_WIDTH]) >= width * 0.78
            and int(components[index, cv2.CC_STAT_HEIGHT]) >= height * 0.78
            and int(components[index, cv2.CC_STAT_AREA])
            / max(1, int(components[index, cv2.CC_STAT_WIDTH]) * int(components[index, cv2.CC_STAT_HEIGHT])) < 0.20
            for index in range(1, count)
        )
        if not frame:
            continue
        minimum = max(500, round(visible_count * 0.035))
        islands = [index for index in range(1, count)
                   if int(components[index, cv2.CC_STAT_AREA]) >= minimum
                   and int(components[index, cv2.CC_STAT_WIDTH]) >= 20
                   and int(components[index, cv2.CC_STAT_HEIGHT]) >= 12
                   and not (int(components[index, cv2.CC_STAT_WIDTH]) >= width * 0.78
                            and int(components[index, cv2.CC_STAT_HEIGHT]) >= height * 0.78)]
        if len(islands) < 2:
            continue
        islands.sort(key=lambda index: int(components[index, cv2.CC_STAT_AREA]), reverse=True)
        assigned = np.zeros((height, width), np.bool_)
        staged: list[tuple[Path, dict]] = []
        write_failed = False
        revision_id = uuid4().hex[:10]
        for index in islands[:12]:
            island = cv2.dilate(np.uint8(labels == index), np.ones((3, 3), np.uint8)) != 0
            island &= (image[:, :, 3] > 0) & ~assigned
            ys, xs = np.where(island)
            if len(xs) < minimum:
                continue
            x1, y1, x2, y2 = max(0, int(xs.min()) - 1), max(0, int(ys.min()) - 1), min(width, int(xs.max()) + 2), min(height, int(ys.max()) + 2)
            part = image[y1:y2, x1:x2].copy()
            part[:, :, 3][~island[y1:y2, x1:x2]] = 0
            path = asset_dir / f"{original_path.stem}_{revision_id}_part_{len(staged) + 1:02d}.png"
            if not cv2.imwrite(str(path), part):
                write_failed = True
                break
            assigned |= island
            child = copy.deepcopy(item)
            child.update({"id": path.stem, "x": float(item.get("x") or 0) + x1,
                          "y": float(item.get("y") or 0) + y1, "width": x2 - x1,
                          "height": y2 - y1, "src": f"/media/assets/{project_id}/{path.name}"})
            child.pop("groupId", None)
            child_metadata = child.setdefault("metadata", {})
            child_metadata.pop("editableTextIds", None)
            child_metadata["partitionedFromResidual"] = item.get("id")
            child_metadata["sourcePixelArea"] = int(components[index, cv2.CC_STAT_AREA])
            staged.append((path, child))
        if write_failed or len(staged) < 2:
            for path, _ in staged:
                path.unlink(missing_ok=True)
            continue
        remainder = image.copy()
        remainder[:, :, 3][assigned] = 0
        remainder_path = asset_dir / f"{original_path.stem}_{revision_id}_remainder.png"
        if not cv2.imwrite(str(remainder_path), remainder):
            for path, _ in staged:
                path.unlink(missing_ok=True)
            continue
        item["src"] = f"/media/assets/{project_id}/{remainder_path.name}"
        metadata["residualPartsCreated"] = len(staged)
        insertion = elements.index(item) + 1
        elements[insertion:insertion] = [child for _, child in staged]
        stats["partitionedResidualAssets"] += 1
        stats["residualPartsCreated"] += len(staged)
    return stats
