import json
import time
from pathlib import Path

import pytest
from app.models.project_store import ProjectStore
from app.services.reconstruction.tasks import AnalysisTasks, TaskConflict
from fastapi.testclient import TestClient


def blocking_worker(root, project_id, page, mode, fallback, events):
    events.put(("stage", "vision"))
    time.sleep(30)
    events.put(("prepared", str(Path(root) / project_id / "candidates" / "never")))


def ready_worker(root, project_id, page, mode, fallback, events):
    events.put(("stage", "validation"))
    events.put(("prepared", str(Path(root) / project_id / "candidates" / "ready")))


def error_worker(root, project_id, page, mode, fallback, events):
    events.put(("error", "AI_UNAVAILABLE"))


def fixture_store(tmp_path):
    store = ProjectStore(tmp_path / "outputs")
    project = store.create("任务回归")
    store.add_image(project["id"], {"id": "one", "path": "unused", "source_url": "/fixture.png"})
    store.save_slide(project["id"], 1, {"elements": [], "userEdit": "keep"})
    return store, project["id"]


def wait_terminal(manager, project_id, task_id):
    until = time.monotonic() + 12
    while time.monotonic() < until:
        task = manager.get(project_id, task_id)
        if task["status"] not in {"running", "publishing"}:
            return task
        time.sleep(.05)
    pytest.fail("task did not terminate")


def test_deadline_stops_worker_and_never_publishes(tmp_path):
    store, project_id = fixture_store(tmp_path)
    published = []
    manager = AnalysisTasks(store, deadline_seconds=.6, worker=blocking_worker, publisher=lambda *args: published.append(args))
    task = manager.submit(project_id, 1, "maximum")
    process = manager.live[task["id"]][0]
    result = wait_terminal(manager, project_id, task["id"])
    assert result["status"] == "timed_out"
    assert not process.is_alive() and not published
    assert store.get_slide(project_id, 1)["userEdit"] == "keep"


def test_cancel_and_duplicate_submission_are_idempotent(tmp_path):
    store, project_id = fixture_store(tmp_path)
    manager = AnalysisTasks(store, worker=blocking_worker)
    task = manager.submit(project_id, 1, "maximum")
    process = manager.live[task["id"]][0]
    assert manager.submit(project_id, 1, "maximum")["id"] == task["id"]
    with pytest.raises(TaskConflict):
        manager.submit(project_id, 1, "fast")
    assert manager.cancel(project_id, task["id"])["status"] == "cancelled"
    assert manager.cancel(project_id, task["id"])["status"] == "cancelled"
    assert not process.is_alive()
    assert store.get_slide(project_id, 1)["userEdit"] == "keep"


def test_success_publishes_exactly_once_and_next_page_can_run(tmp_path):
    store, project_id = fixture_store(tmp_path)
    published = []
    manager = AnalysisTasks(store, worker=ready_worker, publisher=lambda *args: published.append(args))
    first = manager.submit(project_id, 1, "maximum")
    assert wait_terminal(manager, project_id, first["id"])["status"] == "succeeded"
    store.add_image(project_id, {"id": "two", "path": "unused", "source_url": "/two.png"})
    with pytest.raises(TaskConflict):
        manager.submit(project_id, 2, "maximum")
    store.approve_page(project_id, 1)
    second = manager.submit(project_id, 2, "maximum")
    assert wait_terminal(manager, project_id, second["id"])["status"] == "succeeded"
    assert len(published) == 2 and manager.latest(project_id)["page"] == 2


def test_restart_marks_orphaned_state_interrupted(tmp_path):
    store, project_id = fixture_store(tmp_path)
    manager = AnalysisTasks(store)
    task = {"id": "a" * 32, "projectId": project_id, "status": "running", "stage": "vision", "startedAt": time.time()}
    manager._save(task)
    result = AnalysisTasks(store).latest(project_id)
    assert result["status"] == "interrupted"
    assert store.get_slide(project_id, 1)["userEdit"] == "keep"


def test_api_restore_cancel_validation_and_secret_free_errors(monkeypatch, tmp_path):
    import app.main as api
    store, project_id = fixture_store(tmp_path)
    manager = AnalysisTasks(store, worker=error_worker)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "analysis_tasks", manager)
    client = TestClient(api.app)
    base = f"/api/projects/{project_id}"
    assert client.post(base + "/analysis-tasks", json={"page": 0}).status_code == 422
    assert client.post(base + "/analysis-tasks", json={"page": 2}).status_code == 404
    response = client.post(base + "/analysis-tasks", json={"page": 1})
    assert response.status_code == 202
    task = response.json()
    result = wait_terminal(manager, project_id, task["id"])
    assert result["errorCode"] == "AI_UNAVAILABLE"
    restored = client.get(base).json()
    assert restored["task"]["id"] == task["id"]
    assert restored["slides"][0]["userEdit"] == "keep"
    assert "path" not in restored["images"][0]
    assert "apiKey" not in json.dumps(restored)
    assert client.get(base + "/analysis-tasks/bad").status_code == 404


def test_prepare_only_never_publishes(monkeypatch, tmp_path):
    from backend.tests.test_ai_reconstruction_pipeline import fixture_pipeline
    pipeline, project_id, _ = fixture_pipeline(monkeypatch, tmp_path)
    stages = []
    pipeline.analyze_project(project_id, "maximum", prepare_only=True, progress=stages.append)
    root = pipeline.store.root / project_id
    assert not (root / "slides/page_1.json").exists()
    assert (pipeline.prepared_candidate / "publication.json").exists()
    assert stages == ["preprocessing", "ocr", "segmentation", "layout", "text_reconstruction", "validation", "pptx"]
    pipeline.publish_candidate(pipeline.store, pipeline.prepared_candidate)
    assert (root / "slides/page_1.json").exists()
