"""The canonical graph is Layout JSON.elements, not a second mutable scene."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

OWNERS = {"editable_text", "native_shape", "movable_image", "background", "intentional_ignore"}
SHAPES = {"rectangle", "roundedRectangle", "ellipse", "line", "arrow"}


def canonicalize(layout: dict[str, Any]) -> dict[str, Any]:
    """Refresh derived fields after manual edits; keep one authoritative list."""
    for item in layout.get("elements", []):
        kind = item.get("type")
        item["bbox"] = {"left": item.get("x", 0), "top": item.get("y", 0), "width": item.get("width", 0), "height": item.get("height", 0)}
        metadata = item.setdefault("metadata", {})
        if kind == "text" and item.get("lines") and "\n".join(str(line.get("text", "")) for line in item["lines"]) != str(item.get("text") or ""):
            metadata.setdefault("sourceOCRLines", copy.deepcopy(item["lines"]))
            item["lines"] = [{"text": text} for text in str(item.get("text") or "").split("\n")]
            metadata["originalLineCount"] = len(item["lines"])
        if layout.get("sceneVersion") == "3.0":
            owner = "editable_text" if kind == "text" else "native_shape" if kind in SHAPES else "background" if kind == "background" else "movable_image" if kind == "image" else "intentional_ignore"
            item.update(owner=owner, editable=kind not in {"background", "group"}, parent=item.get("groupId"), assetPath=item.get("src"), source=item.get("source") or "manual")
            if owner == "native_shape" and item["source"] == "manual":
                metadata["geometryVerified"] = True
            strategy = {"movable_image": "transparent_image", "background": "background_image", "intentional_ignore": "group"}.get(owner, owner)
            item["reconstructionStrategy"] = strategy
            metadata["reconstructionStrategy"] = strategy
            for key in ("suppressed", "suppressRender", "ownedBy", "preserveWholeAsset", "wholeBadgeAsset"):
                metadata.pop(key, None)
    return layout


def scene_view(layout: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(canonicalize(layout))
    result["canvas"] = result["slide"]
    return result


def resolve_asset(src: str | None, output_root: Path) -> Path | None:
    if not src:
        return None
    parts = src.split("?", 1)[0].replace("\\", "/").split("/")
    if len(parts) == 5 and parts[1] == "media" and parts[2] in {"assets", "backgrounds"}:
        if parts[3] in {".", ".."} or parts[4] in {".", ".."}:
            return None
        return output_root / parts[3] / parts[2] / parts[4]
    path = Path(src)
    return path if path.is_absolute() else None


def valid_replacement(item: dict[str, Any], output_root: Path) -> bool:
    if item.get("owner") not in OWNERS:
        return False
    metadata = item.get("metadata") or {}
    if any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
        return False
    if not all(math.isfinite(float(item.get(key, 0))) for key in ("x", "y", "width", "height")):
        return False
    if float(item.get("width", 0)) <= 0 or float(item.get("height", 0)) <= 0:
        return False
    owner = item["owner"]
    if owner == "editable_text":
        return item.get("type") == "text" and bool(str(item.get("text") or "").strip())
    if owner == "native_shape":
        return item.get("type") in SHAPES and bool(metadata.get("geometryVerified"))
    if owner in {"movable_image", "background"}:
        path = resolve_asset(item.get("src"), output_root)
        if not path or not path.is_file():
            return False
        try:
            with Image.open(path) as asset:
                asset.verify()
            return True
        except (OSError, ValueError):
            return False
    return bool(metadata.get("explicitUserIgnore"))


class OwnershipLedger:
    """Pixel leases granted ONLY after a renderable replacement exists."""
    def __init__(self, shape: tuple[int, int], output_root: Path) -> None:
        self.output_root = output_root
        self.covered = np.zeros(shape, dtype=np.uint8)
        self.claims: dict[str, np.ndarray] = {}

    def claim(self, item: dict[str, Any], mask: np.ndarray) -> None:
        if mask.shape != self.covered.shape or not valid_replacement(item, self.output_root):
            raise ValueError("Deletion denied: no reliable replacement owner")
        mask = (mask > 0).astype(np.uint8) * 255
        self.claims[item["id"]] = mask
        self.covered |= mask

    def authorize(self, requested: np.ndarray) -> np.ndarray:
        return np.where((requested > 0) & (self.covered > 0), 255, 0).astype(np.uint8)
