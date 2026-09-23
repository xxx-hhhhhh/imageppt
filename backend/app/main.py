from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import OUTPUTS_DIR, PUBLIC_SHARED_MODE, TEMP_DIR, UPLOADS_DIR, VISION_PROVIDER, ensure_runtime_dirs
from app.models.project_store import ProjectStore
from app.schemas.layout import AnalyzeResponse, LayoutJSON
from app.schemas.project import ProjectCreate, ProjectResponse, UploadResponse
from app.schemas.vision import VisionSettingsPayload, VisionSettingsResponse, VisionTestResponse
from app.services.pptx import PPTXRenderer
from app.services.reconstruction import ReconstructionPipeline
from app.services.reconstruction.pipeline import AIUnavailableError
from app.services.reconstruction.downgrade import downgrade_problem_regions
from app.services.settings.runtime_settings import load_vision_settings, mask_api_key, save_vision_settings
from app.services.vision.router import VisionRouter
from app.services.vision.proxy import display_proxy_url, proxy_tcp_test, resolve_proxy
from app.services.paddle_runtime import paddle_diagnostics


ensure_runtime_dirs()
store = ProjectStore()
renderer = PPTXRenderer()
app = FastAPI(title="Image2EditablePPT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def project_response(record: dict) -> ProjectResponse:
    return ProjectResponse(id=record["id"], name=record["name"], createdAt=record["createdAt"], imageCount=len(record.get("images", [])))


def _get_project(project_id: str) -> dict:
    try:
        return store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "Image2EditablePPT", **paddle_diagnostics()}


@app.post("/api/projects", response_model=ProjectResponse)
def create_project(payload: ProjectCreate) -> ProjectResponse:
    return project_response(store.create(payload.name))


@app.post("/api/projects/{project_id}/images", response_model=UploadResponse)
async def upload_images(project_id: str, files: list[UploadFile] = File(...)) -> UploadResponse:
    record = _get_project(project_id)
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    image_records = []
    project_upload_dir = UPLOADS_DIR / project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(status_code=415, detail=f"Unsupported image format: {suffix or 'unknown'}")
        file_name = f"{uuid.uuid4().hex}{suffix}"
        destination = project_upload_dir / file_name
        with destination.open("wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                stream.write(chunk)
        image_record = {
            "id": uuid.uuid4().hex,
            "name": upload.filename,
            "path": str(destination),
            "source_url": f"/media/uploads/{project_id}/{file_name}",
            "content_type": upload.content_type or mimetypes.guess_type(file_name)[0],
        }
        store.add_image(project_id, image_record)
        image_records.append(image_record)
    return UploadResponse(project=project_response(store.get(project_id)), images=image_records)


@app.post("/api/projects/{project_id}/analyze", response_model=AnalyzeResponse)
def analyze_project(project_id: str, mode: str = Query("maximum", pattern="^(fast|standard|high_quality|maximum)$"), page: int = Query(1, ge=1), allow_fallback: bool = False) -> AnalyzeResponse:
    record = _get_project(project_id)
    if not record.get("images"):
        raise HTTPException(status_code=400, detail="Upload at least one image before analysis")
    if page > len(record["images"]):
        raise HTTPException(status_code=404, detail="Page not found")
    if page > 1 and page - 1 not in record.get("approvedPages", []):
        raise HTTPException(status_code=409, detail={"code": "PREVIOUS_PAGE_NOT_APPROVED", "message": "请先确认上一页的重建结果。"})
    try:
        pipeline = ReconstructionPipeline(store)
        slides, provider, warnings = pipeline.analyze_project(project_id, mode, page, allow_fallback)
    except AIUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "AI_UNAVAILABLE", "message": str(exc)}) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="页面重建失败，请重试。") from exc
    routing = pipeline.scene_analyzer.vision_routing
    runtime = load_vision_settings()
    vision_warning = next((item for item in warnings if any(name in item for name in ("Qwen", "Vision", "vision"))), None)
    return AnalyzeResponse(project=project_response(record), slides=slides, provider=provider, ocrProvider=provider, visionProvider=routing.get("usedProvider", "none"), aiUsed=bool(routing.get("aiUsed")), visionModel=routing.get("usedModel"), requestedVisionProvider=routing.get("requestedProvider"), fallbackCount=int(routing.get("fallbackCount", 0)), visionWarning=vision_warning, warnings=warnings)


@app.post("/api/projects/{project_id}/pages/{page}/approve")
def approve_page(project_id: str, page: int) -> dict:
    record = _get_project(project_id)
    if page < 1 or page > len(record.get("images", [])):
        raise HTTPException(status_code=404, detail="Page not found")
    try:
        store.approve_page(project_id, page)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail="Analyze this page before approval") from exc
    return {"page": page, "approved": True, "nextPage": page + 1 if page < len(record["images"]) else None}


@app.post("/api/projects/{project_id}/pages/{page}/downgrade", response_model=LayoutJSON)
def downgrade_page(project_id: str, page: int) -> LayoutJSON:
    _get_project(project_id)
    try:
        return LayoutJSON.model_validate(downgrade_problem_regions(store, project_id, page))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/vision/status")
def vision_status() -> dict:
    runtime = load_vision_settings()
    router = VisionRouter(runtime)
    state = router.status()
    return {"provider": state.get("provider", runtime.selected_provider), "configured": state.get("configured", False), "model": state.get("model"), "enabled": runtime.enabled}


@app.post("/api/vision/test")
def vision_test() -> dict:
    runtime = load_vision_settings()
    result = VisionRouter(runtime).test_connection()
    return {**result, "configured": bool(result.get("success")), "error": None if result.get("success") else result.get("message")}


@app.get("/api/settings/vision", response_model=VisionSettingsResponse)
def get_vision_settings() -> VisionSettingsResponse:
    runtime = load_vision_settings()
    return _vision_settings_response(runtime)


@app.put("/api/settings/vision", response_model=VisionSettingsResponse)
def put_vision_settings(payload: VisionSettingsPayload) -> VisionSettingsResponse:
    _require_private_settings_access()
    runtime = save_vision_settings(payload.model_dump(mode="json"))
    return _vision_settings_response(runtime)


@app.post("/api/settings/vision/test", response_model=VisionTestResponse)
def test_vision_settings() -> VisionTestResponse:
    runtime = load_vision_settings()
    result = VisionRouter(runtime).test_connection()
    return VisionTestResponse(
        success=bool(result.get("success")),
        provider=str(result.get("provider", runtime.selected_provider)),
        model=str(result.get("model") or ""),
        message=str(result.get("message", "连接失败")),
        stage=result.get("stage"),
        statusCode=result.get("statusCode"),
        errorType=result.get("errorType"),
        errorCode=result.get("errorCode"),
        connectionPath=result.get("connectionPath"),
        latencyMs=result.get("latencyMs"),
        diagnostics=result.get("diagnostics", {}),
    )


def _require_private_settings_access() -> None:
    if PUBLIC_SHARED_MODE:
        raise HTTPException(status_code=403, detail="公网共享模式下不能修改或测试 AI 设置")


def _vision_settings_response(runtime) -> VisionSettingsResponse:
    config = runtime.providers["qwen"]
    provider_view = {"qwen": {
        "enabled": config.enabled,
        "configured": bool(config.api_key and config.base_url),
        "apiKeyMasked": mask_api_key(config.api_key),
        "baseUrl": config.base_url,
        "model": config.model or "qwen3-vl-flash",
    }}
    selected = VisionRouter(runtime).status()
    system_proxy = resolve_proxy("system")
    active_proxy = resolve_proxy(runtime.proxy_mode, runtime.manual_proxy)
    tcp = proxy_tcp_test(active_proxy.url)
    network = {
        "systemProxy": display_proxy_url(system_proxy.url),
        "proxyUrl": active_proxy.url or "direct",
        "source": active_proxy.source,
        "status": "● 可用" if tcp["status"] in {"PASS", "SKIPPED"} else "○ 不可用",
        "proxyTcp": tcp,
        "autoConfigURL": system_proxy.auto_config_url,
    }
    return VisionSettingsResponse(enabled=runtime.enabled, selectedProvider=runtime.selected_provider, providers=provider_view, configured=bool(selected.get("configured")), mode=runtime.mode, timeout=runtime.timeout, proxyMode=runtime.proxy_mode, manualProxy=runtime.manual_proxy, network=network)


def _friendly_vision_error(error: Exception) -> str:
    code = getattr(error, "status_code", None)
    if code in {401, 403}:
        return "AI API连接失败：请检查 API Key"
    if code == 429:
        return "AI API额度或频率已达到限制，本次已自动使用基础模式。"
    if "timeout" in str(error).lower() or "timed out" in str(error).lower():
        return "AI响应超时，本次已自动使用 OCR/CV 基础模式。"
    return "AI连接失败，请检查 Base URL、模型和网络设置。"


@app.get("/api/projects/{project_id}/slides", response_model=list[LayoutJSON])
def list_slides(project_id: str) -> list[LayoutJSON]:
    _get_project(project_id)
    return [LayoutJSON.model_validate(item) for item in store.list_slides(project_id)]


@app.get("/api/projects/{project_id}/slides/{page}", response_model=LayoutJSON)
def get_slide(project_id: str, page: int) -> LayoutJSON:
    _get_project(project_id)
    try:
        return LayoutJSON.model_validate(store.get_slide(project_id, page))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Slide not found") from exc


@app.put("/api/projects/{project_id}/slides/{page}", response_model=LayoutJSON)
def update_slide(project_id: str, page: int, layout: LayoutJSON) -> LayoutJSON:
    _get_project(project_id)
    store.save_slide(project_id, page, layout.model_dump(mode="json"))
    return layout


@app.post("/api/projects/{project_id}/export/pptx")
def export_pptx(project_id: str) -> dict:
    record = _get_project(project_id)
    if any(page not in record.get("approvedPages", []) for page in range(1, len(record.get("images", [])) + 1)):
        raise HTTPException(status_code=409, detail="请先逐页确认全部重建结果。")
    layouts = store.list_slides(project_id)
    if not layouts:
        raise HTTPException(status_code=400, detail="Analyze images before exporting")
    try:
        output_path, validation = renderer.render_project(project_id, layouts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PPTX export failed: {exc}") from exc
    return {"downloadUrl": f"/api/projects/{project_id}/export/pptx", "fileName": output_path.name, "validation": validation}


@app.get("/api/projects/{project_id}/export/pptx")
def download_pptx(project_id: str) -> FileResponse:
    _get_project(project_id)
    output_path = OUTPUTS_DIR / project_id / "editable.pptx"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="PPTX has not been exported")
    return FileResponse(output_path, filename="editable.pptx", media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")


@app.get("/media/uploads/{project_id}/{file_name}")
def source_media(project_id: str, file_name: str) -> FileResponse:
    return _media_file(UPLOADS_DIR / project_id / file_name)


@app.get("/media/backgrounds/{project_id}/{file_name}")
def background_media(project_id: str, file_name: str) -> FileResponse:
    return _media_file(OUTPUTS_DIR / project_id / "backgrounds" / file_name)


@app.get("/media/assets/{project_id}/{file_name}")
def asset_media(project_id: str, file_name: str) -> FileResponse:
    return _media_file(OUTPUTS_DIR / project_id / "assets" / file_name)


@app.get("/api/projects/{project_id}/artifacts/{file_name}")
def project_artifact(project_id: str, file_name: str) -> FileResponse:
    _get_project(project_id)
    allowed = {"original.png", "background.png", "reconstructed_preview.png", "initial_preview.png", "final_preview.png", "source.png", "clean_background.png", "reconstruction_plan.json", "vision_debug.json", "difference.png", "visual_score.json", "visual_validation.json", "conversion_report.json", "scene_raw.json", "scene_refined.json", "routing.json", "output.pptx"}
    page_artifact = re.fullmatch(r"(?:original|reconstructed_preview|difference|visual_score|visual_validation)_[1-9][0-9]*\.(?:png|json)", file_name)
    if file_name not in allowed and not page_artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return _media_file(OUTPUTS_DIR / project_id / file_name)


def _media_file(path: Path) -> FileResponse:
    resolved = path.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return FileResponse(resolved, media_type=media_type)
