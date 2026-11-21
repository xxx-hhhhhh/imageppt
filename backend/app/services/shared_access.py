"""Browser-session isolation for the separately launched public instance."""
from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from secrets import token_urlsafe
from threading import Lock

from starlette.requests import Request
from starlette.responses import JSONResponse

COOKIE_NAME = "imageppt_session"
MAX_SESSIONS = 256
MAX_BODY_BYTES = 30 * 1024 * 1024


@dataclass
class SharedSession:
    project_id: str | None = None
    lock: Lock = field(default_factory=Lock)


current_session: ContextVar[SharedSession | None] = ContextVar("public_browser_session", default=None)
sessions: dict[str, SharedSession] = {}


async def shared_access(request: Request, call_next):
    parts = request.url.path.strip("/").split("/")
    if parts[0] not in {"api", "media"}:
        return await call_next(request)

    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "请从分享站点内操作。"}, status_code=403)
        length = request.headers.get("content-length")
        if length is None or not length.isdecimal() or len(length) > 10 or int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "单次请求最多支持 30 MB。"}, status_code=413)

    if parts[0] == "media" and (len(parts) != 4 or not re.fullmatch(r"[a-zA-Z0-9_-]+\.(?:png|jpe?g|webp)", parts[-1])):
        return JSONResponse({"detail": "资源不存在。"}, status_code=404)
    project_route = len(parts) >= 3 and (parts[:2] == ["api", "projects"] or parts[0] == "media")
    session = sessions.get(request.cookies.get(COOKIE_NAME, ""))
    if project_route and (session is None or session.project_id != parts[2]):
        return JSONResponse({"detail": "项目不存在。"}, status_code=404)
    if session is None:
        # Initialize on the workspace's create request so parallel status
        # requests from a fresh page cannot overwrite its session cookie.
        if parts == ["api", "projects"] and request.method == "POST":
            if len(sessions) >= MAX_SESSIONS:
                return JSONResponse({"detail": "分享工作区已满，请稍后再试。"}, status_code=503)
            session_id = token_urlsafe(32)
            session = sessions[session_id] = SharedSession()
        elif request.method not in {"GET", "HEAD", "OPTIONS"}:
            return JSONResponse({"detail": "请刷新首页以创建工作区。"}, status_code=409)
        else:
            session_id = ""
    else:
        session_id = ""

    token = current_session.set(session)
    try:
        response = await call_next(request)
    finally:
        current_session.reset(token)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    if session_id:
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
        response.set_cookie(COOKIE_NAME, session_id, httponly=True, secure=secure, samesite="lax")
    return response
