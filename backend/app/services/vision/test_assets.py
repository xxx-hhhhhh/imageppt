from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PIL import Image


def create_test_image() -> Path:
    """Create a tiny closed-handle PNG in the system temp directory.

    ``NamedTemporaryFile`` must not be used here: Windows keeps its handle
    open and another image client cannot reopen the path while it is alive.
    """
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="image2editableppt_"))
        path = temp_dir / "vision_test.png"
        image = Image.new("RGB", (32, 32), "white")
        try:
            image.save(path, format="PNG")
        finally:
            image.close()
        return path
    except OSError as exc:
        if "temp_dir" in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("本地测试图片创建失败") from exc


def cleanup_test_image(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(path.parent, ignore_errors=True)


__all__ = ["create_test_image", "cleanup_test_image"]
