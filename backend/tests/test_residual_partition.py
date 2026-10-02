from pathlib import Path
import copy

import cv2
import numpy as np

from app.services.reconstruction.residual_partition import partition_sparse_residuals


def test_sparse_frame_and_independent_visuals_become_separate_assets(tmp_path: Path) -> None:
    project_id = "a" * 32
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    image = np.zeros((150, 230, 4), np.uint8)
    image[:, :, :3] = (245, 245, 245)
    cv2.rectangle(image, (0, 0), (229, 149), (150, 100, 50, 255), 1)
    image[25:58, 22:62] = (30, 100, 220, 255)
    image[85:118, 145:190] = (50, 180, 40, 255)
    original = image.copy()
    cv2.imwrite(str(asset_dir / "panel.png"), image)
    layout = {"slide": {"width": 300, "height": 200}, "elements": [
        {"id": "panel", "type": "image", "x": 30, "y": 20, "width": 230, "height": 150,
         "src": f"/media/assets/{project_id}/panel.png", "zIndex": 1,
         "metadata": {"layerRole": "residual"}},
    ]}

    stats = partition_sparse_residuals(layout, asset_dir, project_id)

    assert stats == {"partitionedResidualAssets": 1, "residualPartsCreated": 2}
    assert len(layout["elements"]) == 3
    assert len({item["src"] for item in layout["elements"]}) == 3
    assert all((asset_dir / Path(item["src"]).name).is_file() for item in layout["elements"])
    rebuilt = np.zeros_like(original)
    for item in layout["elements"]:
        part = cv2.imread(str(asset_dir / Path(item["src"]).name), cv2.IMREAD_UNCHANGED)
        x, y = int(item["x"] - 30), int(item["y"] - 20)
        region = rebuilt[y:y + part.shape[0], x:x + part.shape[1]]
        visible = part[:, :, 3] > 0
        assert not np.any(visible & (region[:, :, 3] > 0))
        region[visible] = part[visible]
    assert np.array_equal(rebuilt[:, :, 3], original[:, :, 3])
    assert np.array_equal(rebuilt[original[:, :, 3] > 0, :3], original[original[:, :, 3] > 0, :3])


def test_dense_ribbon_stays_one_complete_asset(tmp_path: Path) -> None:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    image = np.full((90, 270, 4), (25, 60, 210, 255), np.uint8)
    cv2.imwrite(str(asset_dir / "ribbon.png"), image)
    layout = {"slide": {"width": 300, "height": 200}, "elements": [
        {"id": "ribbon", "type": "image", "x": 15, "y": 30, "width": 270, "height": 90,
         "src": "/media/assets/demo/ribbon.png", "metadata": {"layerRole": "residual"}},
    ]}

    assert partition_sparse_residuals(layout, asset_dir, "demo")["residualPartsCreated"] == 0
    assert len(layout["elements"]) == 1


def test_new_partition_never_overwrites_previous_valid_assets(tmp_path: Path) -> None:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    image = np.zeros((150, 230, 4), np.uint8)
    cv2.rectangle(image, (0, 0), (229, 149), (150, 100, 50, 255), 1)
    image[25:58, 22:62] = (30, 100, 220, 255)
    image[85:118, 145:190] = (50, 180, 40, 255)
    cv2.imwrite(str(asset_dir / "panel.png"), image)
    original = {"slide": {"width": 300, "height": 200}, "elements": [
        {"id": "panel", "type": "image", "x": 30, "y": 20, "width": 230, "height": 150,
         "src": "/media/assets/demo/panel.png", "metadata": {"layerRole": "residual"}},
    ]}
    first, second = copy.deepcopy(original), copy.deepcopy(original)
    assert partition_sparse_residuals(first, asset_dir, "demo")["residualPartsCreated"] == 2
    previous = {item["src"]: (asset_dir / Path(item["src"]).name).read_bytes()
                for item in first["elements"]}

    assert partition_sparse_residuals(second, asset_dir, "demo")["residualPartsCreated"] == 2
    assert previous.keys().isdisjoint(item["src"] for item in second["elements"])
    assert all((asset_dir / Path(src).name).read_bytes() == payload for src, payload in previous.items())
