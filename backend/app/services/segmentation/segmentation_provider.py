from __future__ import annotations
from app.utils import image_io

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


class SegmentationProvider(ABC):
    name = "base"

    @abstractmethod
    def segment(self, image_path: Path, asset_dir: Path | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError


class OpenCVSegmentationProvider(SegmentationProvider):
    name = "opencv"

    def segment(self, image_path: Path, asset_dir: Path | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        image = image_io.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = cv2.Canny(gray, 80, 180)
        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.dilate(foreground, kernel, iterations=1)
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        objects: list[dict[str, Any]] = []
        for index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:30]):
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < width * height * 0.01 or w < 16 or h < 16 or w > width * 0.92 or h > height * 0.92:
                continue
            mask = np.zeros((h, w), np.uint8)
            shifted = contour - np.array([[x, y]])
            cv2.drawContours(mask, [shifted], -1, 255, -1)
            item: dict[str, Any] = {"id": f"segment_{index + 1:03d}", "bbox": {"left": float(x), "top": float(y), "width": float(w), "height": float(h)}, "confidence": 0.4, "mask": mask.tolist()}
            if asset_dir and project_id:
                asset_dir.mkdir(parents=True, exist_ok=True)
                rgba = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = mask
                path = asset_dir / f"segment_{uuid4().hex[:12]}_{index + 1:03d}.png"
                if not image_io.imwrite(str(path), rgba):
                    continue
                item["alphaCrop"] = f"/media/assets/{project_id}/{path.name}"
            objects.append(item)
        return objects


class OptionalSAM2Provider(OpenCVSegmentationProvider):
    """Local Ultralytics SAM2 inference with honest, non-destructive fallback."""
    name = "sam2"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = None
        self.warnings: list[str] = []

    def segment(self, image_path: Path, asset_dir: Path | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        seeds = super().segment(image_path)
        if not seeds:
            return []
        try:
            from ultralytics import SAM
            if self.model is None:
                self.model = SAM(str(self.model_path))
            image = image_io.imread(str(image_path))
            h, w = image.shape[:2]
            boxes = [[s["bbox"]["left"], s["bbox"]["top"], s["bbox"]["left"]+s["bbox"]["width"], s["bbox"]["top"]+s["bbox"]["height"]] for s in seeds]
            result = self.model.predict(source=image, bboxes=boxes, verbose=False, device=os.getenv("SAM_DEVICE", "cpu"))[0]
            if result.masks is None:
                raise ValueError("No segmentation masks")
            objects = []
            for raw in result.masks.data.cpu().numpy():
                mask = np.uint8(cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > .5)*255
                if np.count_nonzero(mask) < 4:
                    continue
                x, y, cw, ch = cv2.boundingRect(mask)
                if cw*ch >= w*h*.92:
                    continue  # A near-page mask is not an independent object.
                identifier = f"sam2_{uuid4().hex[:12]}"
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                item = {"id": identifier, "bbox": {"left": x, "top": y, "width": cw, "height": ch},
                        "confidence": .75, "source": "ultralytics_sam2", "contour": [c[:, 0, :].tolist() for c in contours]}
                if asset_dir is not None and project_id:
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    path = asset_dir / f"{identifier}.png"
                    rgba = np.dstack((image[y:y+ch, x:x+cw], mask[y:y+ch, x:x+cw]))
                    if not image_io.imwrite(str(path), rgba):
                        raise OSError("SAM asset write failed")
                    item["alphaCrop"] = f"/media/assets/{project_id}/{path.name}"
                objects.append(item)
            if not objects:
                raise ValueError("No usable independent masks")
            self.name = "sam2"
            return objects
        except Exception as exc:  # noqa: BLE001 - optional inference must retain CV fallback
            self.name = "sam2-fallback-opencv"
            self.warnings = [f"SAM2 unavailable ({type(exc).__name__}); retained OpenCV crops instead"]
            return super().segment(image_path, asset_dir, project_id)


def create_segmentation_provider(preferred: str = "auto") -> tuple[SegmentationProvider, list[str]]:
    warnings: list[str] = []
    model = Path(os.getenv("SAM_MODEL_PATH", str(Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "Image2EditablePPT" / "models" / "sam2.1_t.pt")))
    if preferred in {"sam", "sam2"} or (preferred == "auto" and os.getenv("SAM_MODEL_PATH")):
        if model.is_file():
            return OptionalSAM2Provider(model), warnings
        warnings.append("SAM2 model not installed outside the project; using OpenCV segmentation fallback")
    return OpenCVSegmentationProvider(), warnings
