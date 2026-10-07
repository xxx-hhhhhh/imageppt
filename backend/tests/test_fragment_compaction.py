
import cv2
import numpy as np
from app.services.reconstruction.owner_gate import (
    compact_retained_fragments,
    ownership_evidence,
)
from app.services.reconstruction.revision_integrity import _fragment_union_covers


def test_fragment_compaction_has_exact_union_proof_and_retains_old_assets(tmp_path):
    root = tmp_path / "project"
    assets = root / "assets"
    assets.mkdir(parents=True)
    source = np.full((180, 300, 3), 255, np.uint8)
    layout = {"elements": []}
    for index in range(40):
        x, y = 15 + index % 10 * 25, 15 + index // 10 * 35
        source[y:y+3, x:x+3] = (40, 70, 140)
        asset = np.dstack((source[y:y+3, x:x+3], np.full((3, 3), 255, np.uint8)))
        filename = f"old_{index}.png"
        cv2.imwrite(str(assets / filename), asset)
        layout["elements"].append({"id": filename, "type": "image", "x": x, "y": y, "width": 3, "height": 3,
                                  "src": f"/media/assets/project/{filename}", "metadata": {"retainedVisual": True, "reconstructionStrategySource": "owner_gate"}})
    old_files = {p.name: p.read_bytes() for p in assets.glob("*.png")}
    assert compact_retained_fragments(source, layout, assets, "project", 1) == 40
    retired = [item for item in layout["elements"] if item.get("metadata", {}).get("suppressed")]
    new = [item for item in layout["elements"] if item.get("metadata", {}).get("fragmentCluster")]
    assert 0 < len(new) < len(retired)
    assert ownership_evidence(source, layout, assets)[1]["unownedPixelCount"] == 0
    assert all(_fragment_union_covers(root, item, new) for item in retired)
    assert old_files == {name: (assets / name).read_bytes() for name in old_files}
    assert compact_retained_fragments(source, layout, assets, "project", 1) == 0
    # A replacement declaration with empty/missing assets cannot retire owners.
    corrupted = [{**new[0], "src": "/media/assets/project/missing.png"}]
    assert not _fragment_union_covers(root, retired[0], corrupted)
