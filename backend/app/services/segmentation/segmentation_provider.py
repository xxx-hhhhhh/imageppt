from __future__ import annotations

import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = cv2.Canny(gray, 25, 80)
        kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.dilate(foreground, kernel, iterations=1)
        contours, _ = cv2.findContours(foreground, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        height, width = image.shape[:2]
        objects: list[dict[str, Any]] = []
        for index, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:80]):
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < width * height * 0.002 or w < 16 or h < 16 or w > width * 0.92 or h > height * 0.92:
                continue
            mask = np.zeros((h, w), np.uint8)
            shifted = contour - np.array([[x, y]])
            cv2.drawContours(mask, [shifted], -1, 255, -1)
            item: dict[str, Any] = {"id": f"segment_{index + 1:03d}", "bbox": {"left": float(x), "top": float(y), "width": float(w), "height": float(h)}, "confidence": 0.4, "contour": shifted.reshape(-1, 2).tolist(), "source": "opencv"}
            if asset_dir and project_id:
                asset_dir.mkdir(parents=True, exist_ok=True)
                rgba = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = mask
                path = asset_dir / f"segment_{uuid.uuid4().hex}_{index + 1:03d}.png"
                cv2.imencode(".png", rgba)[1].tofile(path)
                item["alphaCrop"] = f"/media/assets/{project_id}/{path.name}"
            objects.append(item)
        return objects


class OptionalSAM2Provider(SegmentationProvider):
    """Real Ultralytics SAM/SAM2 inference, with honest per-call CV fallback.

    No implicit download to the checkout. Set SAM_MODEL_PATH to external weights.
    """
    name = "ultralytics-sam2"

    def __init__(self, model_path: Path) -> None:
        import torch
        from ultralytics import SAM
        torch.set_num_threads(max(1, int(os.getenv("SAM_CPU_THREADS", "4"))))
        self.model = SAM(str(model_path))
        self.fallback = OpenCVSegmentationProvider()
        self.warnings: list[str] = []
        self.prompt_regions: list[dict] = []

    def set_prompt_regions(self, regions: list[dict]) -> None:
        self.prompt_regions = [r for r in regions if r.get("type") not in {"text", "background"} and float(r.get("confidence") or 0) >= 0.6]

    def segment(self, image_path: Path, asset_dir: Path | None = None, project_id: str | None = None) -> list[dict[str, Any]]:
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            # Box prompts encode the image once and bound decoder work on CPU.
            # Segment-everything grids are too slow for an interactive local editor.
            proposals = self.fallback.segment(image_path)
            prompts = []
            for region in self.prompt_regions[:12]:
                box = region.get("bbox") or {}
                try:
                    x, y, w, h = [float(box[key]) for key in ("left", "top", "width", "height")]
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1:
                    x, y, w, h = x*width, y*height, w*width, h*height
                if x >= 0 and y >= 0 and w > 8 and h > 8 and x+w <= width and y+h <= height and w*h < width*height*0.85:
                    prompts.append([x, y, x+w, y+h])
            for p in proposals:
                box = [p["bbox"]["left"], p["bbox"]["top"], p["bbox"]["left"]+p["bbox"]["width"], p["bbox"]["top"]+p["bbox"]["height"]]
                if any(max(abs(a-b) for a, b in zip(box, prior)) < 6 for prior in prompts):
                    continue
                prompts.append(box)
                if len(prompts) == 24:
                    break
            if not prompts:
                self.name = "opencv-fallback"
                return self.fallback.segment(image_path, asset_dir, project_id)
            results = self.model.predict(source=image, bboxes=prompts, imgsz=1024, device=os.getenv("SAM_DEVICE", "cpu"), verbose=False)
            candidates = []
            for result in results:
                if result.masks is None:
                    continue
                for data in result.masks.data.cpu().numpy():
                    mask = (cv2.resize(data.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8) * 255
                    area = np.count_nonzero(mask)
                    if area < 64 or area > width * height * 0.85:
                        continue
                    if any(np.count_nonzero((mask > 0) & (prior > 0)) / max(1, np.count_nonzero((mask > 0) | (prior > 0))) > 0.9 for prior in candidates):
                        continue
                    candidates.append(mask)
            objects = []
            for index, mask in enumerate(sorted(candidates, key=np.count_nonzero, reverse=True)[:80]):
                x, y, w, h = cv2.boundingRect(mask)
                item = {"id": f"sam_{index:03d}", "bbox": {"left": x, "top": y, "width": w, "height": h}, "confidence": 0.75, "source": "ultralytics-sam2"}
                if asset_dir and project_id:
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    crop = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2BGRA)
                    crop[:, :, 3] = mask[y:y+h, x:x+w]
                    path = asset_dir / f"sam_{uuid.uuid4().hex}.png"
                    cv2.imencode(".png", crop)[1].tofile(path)
                    item["alphaCrop"] = f"/media/assets/{project_id}/{path.name}"
                objects.append(item)
            if objects:
                self.name = "ultralytics-sam2"
                return objects
        except Exception as exc:  # noqa: BLE001 -- optional segmentation adapter boundary
            self.warnings.append(f"SAM inference failed ({type(exc).__name__}); retained all visual content through CV/residual assets")
        self.name = "opencv-fallback"
        return self.fallback.segment(image_path, asset_dir, project_id)


def create_segmentation_provider(preferred: str = "auto") -> tuple[SegmentationProvider, list[str]]:
    warnings: list[str] = []
    default_model = Path(os.getenv("LOCALAPPDATA", str(Path.home() / ".cache"))) / "Image2EditablePPT" / "models" / "sam2.1_t.pt"
    model_path = Path(os.getenv("SAM_MODEL_PATH", str(default_model)))
    if preferred in {"auto", "sam", "sam2"} and model_path.is_file():
        try:
            return OptionalSAM2Provider(model_path), warnings
        except Exception:  # noqa: BLE001 -- incompatible optional torch/model must fall back
            warnings.append("SAM runtime unavailable; using OpenCV segmentation fallback")
    elif preferred in {"sam", "sam2", "auto"}:
        warnings.append("SAM weights not configured; using contour segmentation and lossless residual assets")
    return OpenCVSegmentationProvider(), warnings
