from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from app.services.reconstruction.residual_objects import visual_candidate_mask


VISUAL_TYPES = {"image", "rectangle", "roundedRectangle", "ellipse", "line", "arrow"}


def visible_images(layout: dict) -> list[dict]:
    return [item for item in layout.get("elements", []) if item.get("type") == "image" and _visible(item)]


def protected_visuals(layout: dict) -> list[dict]:
    return [item for item in layout.get("elements", []) if item.get("type") in VISUAL_TYPES and _visible(item)]


def _visible(item: dict) -> bool:
    metadata = item.get("metadata") or {}
    return not any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy"))


def asset_path(root: Path, src: str | None) -> Path | None:
    clean = str(src or "").split("?", 1)[0].replace("\\", "/")
    parts = clean.strip("/").split("/")
    if len(parts) == 4 and parts[:2] == ["media", "assets"] and parts[2] == root.name and Path(parts[3]).name == parts[3]:
        return root / "assets" / parts[3]
    direct = Path(clean)
    if direct.is_absolute() and direct.resolve().is_relative_to(root.resolve()):
        return direct
    return None


def localize_project_assets(root: Path, layout: dict) -> int:
    """Copy referenced project assets into this slide's project before saving it.

    Older layouts can point to an asset in another project. Those images render
    until that project is removed, then revisions and exports lose the object.
    """
    changed = 0
    output_root = root.parent.resolve()
    destination = root / "assets"
    for item in layout.get("elements", []):
        if item.get("type") != "image" or asset_path(root, item.get("src")) is not None:
            continue
        clean = str(item.get("src") or "").split("?", 1)[0].replace("\\", "/")
        parts = clean.strip("/").split("/")
        if (len(parts) != 4 or parts[:2] != ["media", "assets"]
                or parts[2] in {"", ".", ".."} or Path(parts[3]).name != parts[3]):
            continue
        source = (output_root / parts[2] / "assets" / parts[3]).resolve()
        if not source.is_relative_to(output_root) or not source.is_file():
            continue
        try:
            with Image.open(source) as image:
                image.verify()
        except (OSError, ValueError):
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        name = f"imported_{digest}_{source.name}"
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / name
        if not target.is_file():
            shutil.copy2(source, target)
        item["src"] = f"/media/assets/{root.name}/{name}"
        item.setdefault("metadata", {})["assetLocalizedFrom"] = clean
        changed += 1
    return changed


def recover_legacy_revision_assets(root: Path, layout: dict) -> int:
    """Repair only URLs written by the old revision asset-root bug."""
    recovered = 0
    for item in visible_images(layout):
        src = str(item.get("src") or "").split("?", 1)[0]
        prefix = "/media/assets/revisions/"
        if not src.startswith(prefix):
            continue
        name = src[len(prefix):]
        if not name or Path(name).name != name:
            continue
        original_name = name.split("_", 2)[-1] if name.startswith("revision_") else name
        original = root / "assets" / original_name
        if original.is_file():
            replacement = original
        else:
            legacy = root / "revisions" / "assets" / name
            if not legacy.is_file():
                continue
            replacement = root / "assets" / f"recovered_{uuid4().hex[:8]}_{name}"
            replacement.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy, replacement)
        try:
            with Image.open(replacement) as image:
                image.verify()
        except (OSError, ValueError):
            continue
        item["src"] = f"/media/assets/{root.name}/{replacement.name}"
        recovered += 1
    return recovered


def inspect_assets(root: Path, layout: dict) -> dict:
    images = visible_images(layout)
    missing: list[str] = []
    for item in images:
        path = asset_path(root, item.get("src"))
        if path is None or not path.is_file():
            missing.append(str(item.get("id")))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            missing.append(str(item.get("id")))
    return {"assets": len(images), "missingAssetCount": len(missing), "missingAssetIds": missing}


def white_area_ratio(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return 1.0
    return float(np.mean(np.all(image >= 245, axis=2)))


def assess_revision(root: Path, baseline: dict, candidate: dict, background_before: Path, background_after: Path, preview_before: Path, preview_after: Path, protected_boxes: list[list[float]] | None = None, source_path: Path | None = None) -> dict:
    before = inspect_assets(root, baseline)
    after = inspect_assets(root, candidate)
    before_by_id = {str(item.get("id")): item for item in visible_images(baseline)}
    after_by_id = {str(item.get("id")): item for item in visible_images(candidate)}
    missing_existing = set(before_by_id) - set(after_by_id)
    candidate_by_id = {str(item.get("id")): item for item in candidate.get("elements", [])}
    declared_replacements: set[str] = set()
    used_replacements: set[str] = set()
    for old_id in sorted(missing_existing):
        retired = candidate_by_id.get(old_id) or {}
        replacements = (retired.get("metadata") or {}).get("replacementOwners") or []
        if replacements and all(key not in before_by_id and key in after_by_id for key in replacements):
            if _fragment_union_covers(root, before_by_id[old_id], [after_by_id[key] for key in replacements]):
                declared_replacements.add(old_id)
                continue
        new_id = str((retired.get("metadata") or {}).get("replacedBy") or "")
        replacement = after_by_id.get(new_id)
        if (new_id in used_replacements or new_id in before_by_id or replacement is None
                or (replacement.get("metadata") or {}).get("replacesAssetId") != old_id):
            continue
        if _old_asset_coverage(before_by_id[old_id], replacement) < 0.9:
            continue
        declared_replacements.add(old_id)
        used_replacements.add(new_id)
    missing_unreplaced = missing_existing - declared_replacements
    before_visuals = {str(item.get("id")): item for item in protected_visuals(baseline)}
    after_visuals = {str(item.get("id")): item for item in protected_visuals(candidate)}
    missing_shapes = {key: item for key, item in before_visuals.items()
                      if item.get("type") != "image"
                      and (key not in after_visuals or after_visuals[key].get("type") != item.get("type"))}
    new_visuals = [item for key, item in after_visuals.items()
                   if key not in before_visuals or before_visuals[key].get("type") != item.get("type")]
    used_shape_replacements: set[str] = set()
    for old_id, old in list(missing_shapes.items()):
        replacement = next((item for item in new_visuals
                            if str(item.get("id")) not in used_shape_replacements
                            and _old_asset_coverage(old, item) >= 0.9), None)
        if replacement is not None:
            used_shape_replacements.add(str(replacement.get("id")))
            del missing_shapes[old_id]
    preserved = sum(before_by_id[key].get("src") == after_by_id[key].get("src") for key in before_by_id.keys() & after_by_id.keys())
    replaced = len(before_by_id.keys() & after_by_id.keys()) - preserved + len(declared_replacements)
    background_before_white = white_area_ratio(background_before)
    background_after_white = white_area_ratio(background_after)
    preview_before_white = white_area_ratio(preview_before)
    preview_after_white = white_area_ratio(preview_after)
    outside_change = _outside_change_ratio(preview_before, preview_after, protected_boxes or [])
    newly_lost, largest_new_loss = _new_source_visual_loss(source_path, preview_before, preview_after, candidate)
    errors = []
    if after["missingAssetCount"]:
        errors.append("missing_assets")
    if missing_unreplaced or (after["assets"] < before["assets"] and not declared_replacements):
        errors.append("lost_existing_images")
    if missing_shapes:
        errors.append("lost_existing_shapes")
    if (background_after_white > background_before_white + 0.035
            and not _separated_background_is_covered(background_before, background_after,
                                                     preview_before, preview_after, new_visuals)):
        errors.append("background_over_whitened")
    if (preview_after_white > preview_before_white + 0.035
            and not _whitening_matches_source(source_path, preview_before, preview_after)):
        errors.append("preview_over_whitened")
    if largest_new_loss >= 24 or newly_lost >= 64:
        errors.append("new_source_visual_loss")
    if outside_change > 0.015:
        errors.append("untargeted_preview_change")
    if len(protected_visuals(candidate)) < len(protected_visuals(baseline)) - len(declared_replacements) - 1:
        errors.append("lost_visual_objects")
    return {
        "assetsBefore": before["assets"], "assetsAfter": after["assets"],
        "missingAssetCount": after["missingAssetCount"], "missingAssetIds": after["missingAssetIds"],
        "preservedAssetCount": preserved, "replacedAssetCount": replaced,
        "protectedRegions": len(protected_visuals(baseline)),
        "backgroundWhiteBefore": round(background_before_white, 4), "backgroundWhiteAfter": round(background_after_white, 4),
        "previewWhiteBefore": round(preview_before_white, 4), "previewWhiteAfter": round(preview_after_white, 4),
        "outsideTargetChangeRatio": round(outside_change, 4),
        "newlyLostVisualPixels": newly_lost, "largestNewVisualLoss": largest_new_loss,
        "integrityErrors": errors,
    }


def _fragment_union_covers(root: Path, old: dict, replacements: list[dict]) -> bool:
    """A many-to-fewer asset count change needs pixel proof, not declarations."""
    path = asset_path(root, old.get("src"))
    original = cv2.imread(str(path), -1) if path and path.is_file() else None
    if original is None or original.ndim != 3 or original.shape[2] != 4 or old.get("rotation", 0) or old.get("crop"):
        return False
    if original.shape[:2] != (round(float(old["height"])), round(float(old["width"]))):
        return False
    x, y = round(float(old["x"])), round(float(old["y"]))
    height, width = original.shape[:2]
    evidence = np.zeros((height, width), bool)
    for replacement in replacements:
        path = asset_path(root, replacement.get("src"))
        asset = cv2.imread(str(path), -1) if path and path.is_file() else None
        if asset is None or asset.ndim != 3 or asset.shape[2] != 4 or replacement.get("rotation", 0) or replacement.get("crop"):
            return False
        ax, ay = round(float(replacement["x"])), round(float(replacement["y"]))
        if asset.shape[:2] != (round(float(replacement["height"])), round(float(replacement["width"]))):
            return False
        left, top = max(x,ax), max(y,ay)
        right, bottom = min(x+width, ax+asset.shape[1]), min(y+height, ay+asset.shape[0])
        if right <= left or bottom <= top:
            continue
        old_patch = original[top-y:bottom-y, left-x:right-x]
        new_patch = asset[top-ay:bottom-ay, left-ax:right-ax]
        evidence[top-y:bottom-y, left-x:right-x] |= ((new_patch[:, :, 3] >= old_patch[:, :, 3])
            & np.all(new_patch[:, :, :3] == old_patch[:, :, :3], axis=2))
    return bool(np.any(original[:, :, 3] > 0) and np.all(evidence[original[:, :, 3] > 0]))


def _new_source_visual_loss(source_path: Path | None, preview_before: Path, preview_after: Path,
                            candidate: dict) -> tuple[int, int]:
    """Catch a newly erased local object even when other fixes improve page totals."""
    if source_path is None:
        return 0, 0
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    before = cv2.imread(str(preview_before), cv2.IMREAD_COLOR)
    after = cv2.imread(str(preview_after), cv2.IMREAD_COLOR)
    if source is None or before is None or after is None or source.shape != before.shape or source.shape != after.shape:
        return 0, 0
    candidate_mask = visual_candidate_mask(source)
    source_int = source.astype(np.int16)
    before_erased = np.all(before >= 253, axis=2) & (np.max(np.abs(source_int - before.astype(np.int16)), axis=2) >= 2)
    after_erased = np.all(after >= 253, axis=2) & (np.max(np.abs(source_int - after.astype(np.int16)), axis=2) >= 2)
    newly_lost = np.uint8(candidate_mask & after_erased & ~before_erased)
    height, width = newly_lost.shape
    for item in candidate.get("elements", []):
        metadata = item.get("metadata") or {}
        if item.get("type") != "text" or any(metadata.get(key) for key in ("suppressed", "suppressRender", "ownedBy")):
            continue
        raw = metadata.get("rawOCRBBox")
        if isinstance(raw, list) and len(raw) == 4:
            x1, y1, x2, y2 = (int(round(float(value))) for value in raw)
        else:
            x1, y1 = int(round(float(item.get("x") or 0))), int(round(float(item.get("y") or 0)))
            x2 = x1 + int(round(float(item.get("width") or 0)))
            y2 = y1 + int(round(float(item.get("height") or 0)))
        x1, y1, x2, y2 = max(0, x1 - 2), max(0, y1 - 2), min(width, x2 + 2), min(height, y2 + 2)
        if x2 > x1 and y2 > y1:
            # OCR boxes often enclose a badge or its colored support. Ignore
            # source pixels that plausibly belong to the editable glyphs,
            # rather than hiding every visual pixel in the whole textbox.
            color = str((item.get("style") or {}).get("color") or "#111827").lstrip("#")
            try:
                glyph_bgr = np.frombuffer(bytes.fromhex(color)[::-1], dtype=np.uint8).astype(np.int16)
            except ValueError:
                continue
            if len(glyph_bgr) != 3:
                continue
            patch = source_int[y1:y2, x1:x2]
            near_glyph = np.max(np.abs(patch - glyph_bgr), axis=2) <= 55
            newly_lost[y1:y2, x1:x2][near_glyph] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(newly_lost, 8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
    return int(np.count_nonzero(newly_lost)), largest


def _whitening_matches_source(source_path: Path | None, before_path: Path, after_path: Path) -> bool:
    """Allow removing a false colored plate when the source itself is white."""
    if source_path is None:
        return False
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    if source is None or before is None or after is None or source.shape != before.shape or source.shape != after.shape:
        return False
    newly_white = np.all(after >= 245, axis=2) & ~np.all(before >= 245, axis=2)
    if not np.any(newly_white):
        return False
    source_error = np.max(np.abs(source.astype(np.int16) - after.astype(np.int16)), axis=2)
    return float(np.mean(source_error[newly_white] <= 16)) >= 0.98


def _separated_background_is_covered(background_before: Path, background_after: Path,
                                     preview_before: Path, preview_after: Path,
                                     new_visuals: list[dict]) -> bool:
    """Allow a white base only when a new object preserves the removed pixels."""
    if not new_visuals:
        return False
    before = cv2.imread(str(background_before), cv2.IMREAD_COLOR)
    after = cv2.imread(str(background_after), cv2.IMREAD_COLOR)
    old_preview = cv2.imread(str(preview_before), cv2.IMREAD_COLOR)
    new_preview = cv2.imread(str(preview_after), cv2.IMREAD_COLOR)
    if any(image is None for image in (before, after, old_preview, new_preview)) or not all(
            image.shape == before.shape for image in (after, old_preview, new_preview)):
        return False
    height, width = before.shape[:2]
    removed = np.any(before < 245, axis=2) & np.all(after >= 245, axis=2)
    if not np.any(removed):
        return False
    covered = np.zeros((height, width), np.bool_)
    for item in new_visuals:
        x1 = max(0, round(float(item.get("x") or 0)))
        y1 = max(0, round(float(item.get("y") or 0)))
        x2 = min(width, round(float(item.get("x") or 0) + float(item.get("width") or 0)))
        y2 = min(height, round(float(item.get("y") or 0) + float(item.get("height") or 0)))
        if x2 > x1 and y2 > y1:
            covered[y1:y2, x1:x2] = True
    if float(np.mean(covered[removed])) < 0.98:
        return False
    error = np.max(np.abs(old_preview.astype(np.int16) - new_preview.astype(np.int16)), axis=2)
    return float(np.mean(error[removed] <= 16)) >= 0.98


def _old_asset_coverage(old: dict, new: dict) -> float:
    x1, y1 = float(old.get("x") or 0), float(old.get("y") or 0)
    x2, y2 = x1 + float(old.get("width") or 0), y1 + float(old.get("height") or 0)
    nx1, ny1 = float(new.get("x") or 0), float(new.get("y") or 0)
    nx2, ny2 = nx1 + float(new.get("width") or 0), ny1 + float(new.get("height") or 0)
    overlap = max(0.0, min(x2, nx2) - max(x1, nx1)) * max(0.0, min(y2, ny2) - max(y1, ny1))
    return overlap / max(1.0, (x2 - x1) * (y2 - y1))


def _outside_change_ratio(before_path: Path, after_path: Path, target_boxes: list[list[float]]) -> float:
    before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    if before is None or after is None or before.shape != after.shape:
        return 1.0
    height, width = before.shape[:2]
    allowed = np.zeros((height, width), dtype=np.uint8)
    for box in target_boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        x1, y1, x2, y2 = max(0, x1 - 8), max(0, y1 - 8), min(width, x2 + 8), min(height, y2 + 8)
        if x2 > x1 and y2 > y1:
            allowed[y1:y2, x1:x2] = 1
    changed = np.max(cv2.absdiff(before, after), axis=2) > 16
    return float(np.mean(changed & (allowed == 0)))
