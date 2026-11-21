import app.main as api
from app import config
from app.models.project_store import ProjectStore
from app.services.settings import runtime_settings
from app.services.shared_access import shared_access
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_share_opens_empty_workspaces_and_keeps_owner_key_server_side(monkeypatch, tmp_path):
    from app.services import shared_access as access

    monkeypatch.setattr(access, "sessions", {})
    monkeypatch.setattr(config, "PUBLIC_SHARED_MODE", True)
    monkeypatch.setattr(api, "PUBLIC_SHARED_MODE", True)
    store = ProjectStore(tmp_path)
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "analysis_tasks", api.AnalysisTasks(store))
    monkeypatch.setenv("IMAGE2EDITABLEPPT_SHARE_QWEN", "1")
    local = runtime_settings.VisionSettings(
        providers={"qwen": runtime_settings.ProviderSettings(
            enabled=True, api_key="fixture-owner-key", base_url=runtime_settings.QWEN_BASE_URL, model="fixture",
        )},
    )
    monkeypatch.setattr(runtime_settings, "load_local_vision_settings", lambda: local)
    app = FastAPI()
    app.include_router(api.app.router)
    app.middleware("http")(shared_access)

    a = TestClient(app, base_url="https://share.example.test")
    b = TestClient(app, base_url="https://share.example.test")
    created = a.post("/api/projects", json={})
    assert created.status_code == 200
    assert "HttpOnly" in created.headers["set-cookie"]
    assert "Secure" in created.headers["set-cookie"]
    project = created.json()
    assert project["imageCount"] == 0
    assert b.post("/api/projects", json={}).json()["id"] != project["id"]
    path = f"/api/projects/{project['id']}"
    assert a.get(path).json()["images"] == []
    assert b.get(path).status_code == 404
    assert b.get(f"/media/assets/{project['id']}/sample.png").status_code == 404
    settings = a.get("/api/settings/vision")
    assert settings.json()["publicShared"] and settings.json()["configured"]
    assert settings.json()["providers"]["qwen"]["apiKeyMasked"] == ""
    assert "fixture-owner-key" not in settings.text
    assert a.put("/api/settings/vision", json={}).status_code == 403
