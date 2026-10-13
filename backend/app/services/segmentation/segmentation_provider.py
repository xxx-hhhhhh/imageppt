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
    text_regions: list[dict] = []

    def _text_dominated(self, x: int, y: int, w: int, h: int) -> bool:
        covered = np.zeros((h,w),bool)
        for item in self.text_regions:
            if float(item.get('confidence') or 0)<.9:
                continue
            left,top = max(x,round(item['x'])),max(y,round(item['y']))
            right,bottom = min(x+w,round(item['x']+item['width'])),min(y+h,round(item['y']+item['height']))
            if right>left and bottom>top:
                covered[top-y:bottom-y,left-x:right-x] = True
        return float(covered.mean())>.65

    def segment(self, image_path: Path, asset_dir: Path | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        image = image_io.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = cv2.Canny(gray, 80, 180)
        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.dilate(foreground, kernel, iterations=1)
        # External-only contours stop at the card boundary and never propose
        # the illustration/chart INSIDE that card to SAM.
        contours, _ = cv2.findContours(foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        objects: list[dict[str, Any]] = []
        boxes = []
        for index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
            x, y, w, h = cv2.boundingRect(contour)
            if w*h < width*height*.0015 or min(w,h)<24 or max(w/h,h/w)>10 or w>width*.92 or h>height*.92:
                continue
            if self._text_dominated(x,y,w,h):
                continue  # OCR glyphs are not independent visual-object prompts.
            if any((max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))) /
                   max(1,w*h+bw*bh-max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by)))>.88
                   for bx,by,bw,bh in boxes):
                continue
            boxes.append((x,y,w,h))
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
            if len(objects)>=64:
                break
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
            accepted_masks = []
            for raw in result.masks.data.cpu().numpy():
                mask = np.uint8(cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > .5)*255
                if np.count_nonzero(mask) < 4:
                    continue
                x, y, cw, ch = cv2.boundingRect(mask)
                if cw*ch >= w*h*.92:
                    continue  # A near-page mask is not an independent object.
                if self._text_dominated(x,y,cw,ch):
                    continue
                binary = mask>0
                if any(np.count_nonzero(binary & existing)/max(1,np.count_nonzero(binary | existing))>.80
                       for existing in accepted_masks):
                    continue  # Different nested CV prompts can return ONE SAM object.
                accepted_masks.append(binary)
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
