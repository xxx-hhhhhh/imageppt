"""Regenerate the deterministic complex-module regression image."""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    image = np.full((360, 640, 3), (250, 250, 250), np.uint8)
    cv2.rectangle(image, (38, 55), (290, 196), (242, 245, 247), -1)
    cv2.circle(image, (92, 104), 32, (52, 91, 204), -1)
    cv2.line(image, (77, 104), (107, 104), (255, 255, 255), 6)
    cv2.line(image, (92, 89), (92, 119), (255, 255, 255), 6)
    cv2.putText(image, "MODULE A", (137, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (31, 39, 62), 2)
    cv2.putText(image, "Editable text", (61, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (75, 82, 103), 1)
    cv2.rectangle(image, (335, 76), (594, 263), (246, 242, 236), -1)
    cv2.putText(image, "MODULE B", (363, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (39, 48, 68), 2)
    cv2.arrowedLine(image, (379, 186), (543, 186), (169, 94, 51), 10, tipLength=0.18)
    cv2.putText(image, "Details", (371, 231), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (73, 84, 102), 1)
    cv2.imwrite(str(Path(__file__).with_name("complex_modules.png")), image)


if __name__ == "__main__":
    main()
