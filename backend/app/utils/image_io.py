"""OpenCV pixels with Python filesystem I/O (Windows Unicode/Drive paths).

OpenCV 4.10's filename APIs can fail even when Python/Pillow open the file.
Decode/encode bytes without changing BGR/BGRA, alpha, depth or return semantics.
"""
from __future__ import annotations

from os import PathLike
from pathlib import Path

import cv2
import numpy as np


def imread(filename: str | PathLike[str], flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        encoded = np.fromfile(filename, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def imwrite(filename: str | PathLike[str], image: np.ndarray, params: list[int] | None = None) -> bool:
    # Let OpenCV validate image/format/codec parameters, as imwrite did. Only
    # filesystem failures become False, so the owner gate refuses deletion.
    success, encoded = cv2.imencode(Path(filename).suffix, image, params or [])
    if not success:
        return False
    try:
        encoded.tofile(filename)
    except (OSError, ValueError):
        return False
    return True
