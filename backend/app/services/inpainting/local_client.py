from __future__ import annotations
from app.utils import image_io

import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image


class LocalInpaintUnavailable(RuntimeError):
    pass


class LocalIOPaintClient:
    """IOPaint adapter; discover its JSON endpoint from the running OpenAPI spec."""

    name = "local_lama"

    def __init__(self, url: str, model: str = "lama", timeout: int = 90) -> None:
        parsed = urlparse(url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LOCAL_INPAINT_URL must be an HTTP URL")
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.endpoint: str | None = None
        self.attempts = 0
        self.successes = 0
        self.failures = 0
        self.failed = False

    def probe(self) -> dict[str, str | bool]:
        try:
            response = requests.get(f"{self.url}/openapi.json", timeout=3)
            response.raise_for_status()
            spec = response.json()
            paths = spec.get("paths") or {}
            matching = []
            for path, methods in paths.items():
                operation = methods.get("post") if isinstance(methods, dict) else None
                content = ((operation or {}).get("requestBody") or {}).get("content") or {}
                if "inpaint" in path.lower() and "application/json" in content:
                    matching.append(path)
            if len(matching) != 1:
                raise LocalInpaintUnavailable("OpenAPI did not identify one JSON inpaint route")
            self.endpoint = matching[0]
            request_schema = paths[self.endpoint]["post"]["requestBody"]["content"]["application/json"]["schema"]
            reference = request_schema.get("$ref", "")
            if reference.startswith("#/components/schemas/"):
                request_schema = ((spec.get("components") or {}).get("schemas") or {}).get(reference.rsplit("/", 1)[-1], {})
            if not {"image", "mask"}.issubset(request_schema.get("properties") or {}):
                raise LocalInpaintUnavailable("IOPaint JSON schema has no image and mask fields")
            model_paths = [path for path, methods in paths.items() if path.rstrip("/").endswith("/model") and "get" in methods]
            if model_paths:
                model_response = requests.get(f"{self.url}{model_paths[0]}", timeout=3)
                model_response.raise_for_status()
                actual_model = str(model_response.json().get("name") or "")
                if actual_model.casefold() != self.model.casefold():
                    raise LocalInpaintUnavailable(f"IOPaint model is {actual_model or 'unknown'}, expected {self.model}")
            return {"connected": True, "route": self.endpoint, "model": self.model}
        except (requests.RequestException, ValueError, KeyError, TypeError, LocalInpaintUnavailable) as exc:
            self.endpoint = None
            return {"connected": False, "route": "", "model": self.model, "reason": str(exc)}

    def inpaint(self, image_path: Path, mask: np.ndarray, output_path: Path) -> Path:
        self.attempts += 1
        try:
            if self.failed:
                raise LocalInpaintUnavailable("IOPaint failed earlier in this page")
            if self.endpoint is None and not self.probe()["connected"]:
                raise LocalInpaintUnavailable("IOPaint is not available")
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            if mask.shape != (image.height, image.width):
                raise ValueError("Inpainting mask size does not match source image")
            mask_image = Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8), mode="L")
            image_bytes, mask_bytes = BytesIO(), BytesIO()
            image.save(image_bytes, format="PNG")
            mask_image.save(mask_bytes, format="PNG")
            payload = {"image": base64.b64encode(image_bytes.getvalue()).decode("ascii"), "mask": base64.b64encode(mask_bytes.getvalue()).decode("ascii")}
            response = requests.post(f"{self.url}{self.endpoint}", json=payload, headers={"accept": "image/png"}, timeout=self.timeout)
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as edited:
                if edited.size != image.size:
                    raise ValueError("IOPaint output dimensions changed")
                result = edited.convert("RGB")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(output_path, format="PNG")
            self.successes += 1
            return output_path
        except Exception:
            self.failures += 1
            self.failed = True
            raise

    def inpaint_array(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from tempfile import TemporaryDirectory

        if image.shape[:2] != mask.shape:
            raise ValueError("Inpainting mask size does not match source image")
        with TemporaryDirectory(prefix="imageppt-local-lama-") as directory:
            source, target = Path(directory) / "source.png", Path(directory) / "result.png"
            image_io.imwrite(str(source), image)
            self.inpaint(source, mask, target)
            result = image_io.imread(str(target), cv2.IMREAD_COLOR)
            if result is None or result.shape != image.shape:
                raise ValueError("IOPaint output is invalid")
            return result
