"""Probe and call the configured local IOPaint service without using any vision API."""

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import LOCAL_INPAINT_MODEL, LOCAL_INPAINT_URL, TEMP_DIR  # noqa: E402
from app.services.inpainting.local_client import LocalIOPaintClient  # noqa: E402


def main() -> int:
    client = LocalIOPaintClient(LOCAL_INPAINT_URL, LOCAL_INPAINT_MODEL)
    status = client.probe()
    print(f"connected={status['connected']} route={status['route']} model={status['model']}")
    if not status["connected"]:
        return 1
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    source, target = TEMP_DIR / "lama_verify_input.png", TEMP_DIR / "lama_verify_output.png"
    image = np.full((96, 128, 3), 240, np.uint8)
    cv2.putText(image, "TEST", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(str(source), image)
    mask = np.zeros((96, 128), np.uint8)
    mask[25:64, 15:115] = 255
    client.inpaint(source, mask, target)
    print(f"successes={client.successes} output={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
