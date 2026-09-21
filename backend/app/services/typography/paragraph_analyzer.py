from __future__ import annotations

from typing import Any


def group_lines_into_paragraphs(lines: list[dict[str, Any]], max_vertical_gap: float | None = None) -> list[dict[str, Any]]:
    ordered = sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    paragraphs: list[dict[str, Any]] = []
    for line in ordered:
        if not paragraphs:
            paragraphs.append({"id": f"paragraph_{len(paragraphs) + 1:03d}", "lines": [line]})
            continue
        previous = paragraphs[-1]["lines"][-1]
        previous_height = previous["bbox"][3] - previous["bbox"][1]
        gap = line["bbox"][1] - previous["bbox"][3]
        threshold = max_vertical_gap if max_vertical_gap is not None else max(6.0, previous_height * 0.75)
        overlap_x = min(previous["bbox"][2], line["bbox"][2]) - max(previous["bbox"][0], line["bbox"][0])
        if gap <= threshold and overlap_x > -previous_height * 2:
            paragraphs[-1]["lines"].append(line)
        else:
            paragraphs.append({"id": f"paragraph_{len(paragraphs) + 1:03d}", "lines": [line]})
    for paragraph in paragraphs:
        paragraph["text"] = "\n".join(item["text"] for item in paragraph["lines"])
        xs = [item["bbox"][0] for item in paragraph["lines"]]
        ys = [item["bbox"][1] for item in paragraph["lines"]]
        x2 = [item["bbox"][2] for item in paragraph["lines"]]
        y2 = [item["bbox"][3] for item in paragraph["lines"]]
        paragraph["bbox"] = [min(xs), min(ys), max(x2), max(y2)]
    return paragraphs
