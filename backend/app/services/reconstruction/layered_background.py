from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.services.pptx.renderer import _path_from_src
from app.services.reconstruction.owner_gate import ownership_evidence
if TYPE_CHECKING:
    from app.services.inpainting.provider import InpaintingProvider


def separate_foreground(background_path: Path, elements: list[dict], *, professional_provider: "InpaintingProvider | None" = None, repair_report: list[dict] | None = None) -> int:
    """Remove independently rendered pixels from the slide's background layer.

    The visual asset is left unchanged. Its position therefore renders exactly
    as before, while moving it no longer reveals a second copy underneath.
    """
    background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(background_path)
    height, width = background.shape[:2]
    regions: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for element in elements:
        metadata = element.get("metadata") or {}
        if element.get("type") == "background" or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        strategy = metadata.get("reconstructionStrategy")
        independent_image = element.get("type") == "image" and (metadata.get("preserveWholeAsset") or strategy == "cutout_image")
        planned_shape = strategy == "native_shape" and metadata.get("reconstructionStrategySource") == "planner"
        if not independent_image and not planned_shape:
            continue
        x1, y1 = int(round(float(element.get("x") or 0))), int(round(float(element.get("y") or 0)))
        x2 = x1 + int(round(float(element.get("width") or 0)))
        y2 = y1 + int(round(float(element.get("height") or 0)))
        box = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        if (box[2] - box[0]) * (box[3] - box[1]) > width * height * 0.90:
            continue
        proof_item = dict(element)
        if independent_image:
            actual_path = _path_from_src(element.get("src"))
            if actual_path is None or not actual_path.is_file():
                continue
            proof_item["src"] = str(actual_path)
        evidence, _ = ownership_evidence(background, {"elements": [proof_item]},
                                        background_path.parent.parent / "assets", include_background=False)
        owned = np.uint8(evidence[box[1]:box[3], box[0]:box[2]]) * 255
        if not np.any(owned):
            continue
        regions.append((box, owned))
        metadata["backgroundSeparated"] = True
        metadata["layerRole"] = "cutout_image" if independent_image else "native_shape"
    if not regions:
        return 0
    mask = np.zeros((height, width), dtype=np.uint8)
    for (x1, y1, x2, y2), owned in regions:
        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], owned)
    # Telea handles small icons well. Large photos/charts get a smooth fill
    # sampled outside the owned region so the operation stays bounded.
    small = np.zeros_like(mask)
    for (x1, y1, x2, y2), owned in regions:
        ring = _outer_ring(background, mask, (x1, y1, x2, y2))
        textured = ring.size > 0 and float(np.mean(np.std(ring.astype(np.float32), axis=0))) > 22
        if textured or (professional_provider is not None and professional_provider.name == "local_lama"):
            success = False
            if professional_provider is not None:
                try:
                    with TemporaryDirectory(prefix="imageppt-asset-inpaint-") as workspace:
                        source_file, result_file = Path(workspace) / "source.png", Path(workspace) / "result.png"
                        cv2.imwrite(str(source_file), background)
                        region_mask = np.zeros((height, width), dtype=np.uint8)
                        region_mask[y1:y2, x1:x2] = owned
                        professional_provider.inpaint(source_file, region_mask, result_file)
                        edited = cv2.imread(str(result_file), cv2.IMREAD_COLOR)
                        if edited is not None and edited.shape == background.shape:
                            patch = background[y1:y2, x1:x2]
                            patch[owned > 0] = edited[y1:y2, x1:x2][owned > 0]
                            success = True
                except Exception:
                    pass
            if repair_report is not None:
                problem = "professionalInpaintingApplied" if success else "localInpaintingFallback" if professional_provider is not None and professional_provider.name == "local_lama" else "professionalInpaintingPending"
                repair_report.append({"bbox": [x1, y1, x2, y2], "problem": problem, "provider": professional_provider.name if professional_provider else "unavailable"})
            if success:
                continue
        if (x2 - x1) * (y2 - y1) <= width * height * 0.025:
            small[y1:y2, x1:x2] = np.maximum(small[y1:y2, x1:x2], owned)
        elif ring.size:
            patch = background[y1:y2, x1:x2]
            patch[owned > 0] = np.median(ring, axis=0).astype(np.uint8)
    if np.any(small):
        background = cv2.inpaint(background, small, 4, cv2.INPAINT_TELEA)
    cv2.imwrite(str(background_path), background)
    return len(regions)


def _outer_ring(image: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    height, width = mask.shape
    pad = max(6, min(24, min(x2 - x1, y2 - y1) // 6))
    left, top, right, bottom = max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)
    surround = image[top:bottom, left:right]
    available = mask[top:bottom, left:right] == 0
    return surround[available] if np.any(available) else np.empty((0, 3), dtype=np.uint8)
