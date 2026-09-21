from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class ProxyInfo:
    url: str | None
    source: str
    enabled: bool
    auto_config_url: str | None = None
    error: str | None = None


def read_windows_proxy() -> ProxyInfo:
    """Read the current HKCU Internet Settings values on every call."""
    if not sys.platform.startswith("win"):
        return ProxyInfo(None, "non-windows", False)
    try:
        import winreg

        path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
            enabled = int(_registry_value(key, "ProxyEnable", 0) or 0) == 1
            proxy_server = str(_registry_value(key, "ProxyServer", "") or "").strip()
            auto_config_url = str(_registry_value(key, "AutoConfigURL", "") or "").strip() or None
        if enabled and proxy_server:
            return ProxyInfo(_normalize_proxy(proxy_server), "windows-system", True, auto_config_url)
        # PAC evaluation is intentionally not guessed. The URL is exposed for
        # diagnostics, while environment fallback remains deterministic.
        return ProxyInfo(None, "windows-system", False, auto_config_url)
    except (OSError, FileNotFoundError, ImportError) as exc:
        return ProxyInfo(None, "windows-system", False, error=type(exc).__name__)


def resolve_proxy(proxy_mode: str = "system", manual_proxy: str = "") -> ProxyInfo:
    """Resolve proxy as manual -> Windows -> HTTPS_PROXY -> HTTP_PROXY -> direct."""
    mode = (proxy_mode or "system").lower()
    if mode in {"none", "direct", "off"}:
        return ProxyInfo(None, "direct", False)
    if mode == "manual":
        if manual_proxy.strip():
            return ProxyInfo(_normalize_proxy(manual_proxy.strip()), "manual", True)
        return ProxyInfo(None, "direct", False, error="manual proxy is empty")
    windows = read_windows_proxy()
    if windows.url:
        return windows
    for name in ("HTTPS_PROXY", "HTTP_PROXY"):
        value = os.getenv(name, "").strip()
        if value:
            return ProxyInfo(_normalize_proxy(value), name, True)
    return ProxyInfo(None, "direct", False, auto_config_url=windows.auto_config_url, error=windows.error)


def proxy_tcp_test(proxy_url: str | None, timeout: float = 3.0) -> dict[str, str | bool | None]:
    if not proxy_url:
        return {"status": "SKIPPED", "message": "未使用代理，直接连接", "host": None, "port": None}
    parsed = urlparse(proxy_url)
    if not parsed.hostname or not parsed.port:
        return {"status": "FAIL", "message": "代理地址格式错误", "host": parsed.hostname, "port": parsed.port}
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout):
            return {"status": "PASS", "message": "本地代理端口可用", "host": parsed.hostname, "port": parsed.port}
    except OSError:
        return {"status": "FAIL", "message": "当前系统代理端口不可用，请检查代理软件。", "host": parsed.hostname, "port": parsed.port}


def _registry_value(key, name: str, default):
    try:
        import winreg

        return winreg.QueryValueEx(key, name)[0]
    except FileNotFoundError:
        return default


def _normalize_proxy(value: str) -> str:
    value = value.strip()
    # Windows ProxyServer may contain protocol-specific entries.
    if ";" in value:
        entries = dict(item.split("=", 1) for item in value.split(";") if "=" in item)
        value = entries.get("https") or entries.get("http") or next(iter(entries.values()), value)
    if not value.startswith(("http://", "https://", "socks5://")):
        value = "http://" + value
    return value.rstrip("/")


def display_proxy_url(value: str | None) -> str:
    if not value:
        return "直连"
    parsed = urlparse(value)
    return f"{parsed.hostname}:{parsed.port}" if parsed.hostname and parsed.port else value


__all__ = ["ProxyInfo", "display_proxy_url", "proxy_tcp_test", "read_windows_proxy", "resolve_proxy"]
