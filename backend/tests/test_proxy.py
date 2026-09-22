from __future__ import annotations

from app.services.vision import proxy


def test_proxy_priority_manual_then_windows_then_environment(monkeypatch):
    monkeypatch.setattr(proxy, "read_windows_proxy", lambda: proxy.ProxyInfo("http://127.0.0.1:7897", "windows-system", True))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8443")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8080")

    assert proxy.resolve_proxy("manual", "127.0.0.1:9000").url == "http://127.0.0.1:9000"
    assert proxy.resolve_proxy("system").url == "http://127.0.0.1:7897"

    monkeypatch.setattr(proxy, "read_windows_proxy", lambda: proxy.ProxyInfo(None, "windows-system", False))
    assert proxy.resolve_proxy("system").url == "http://127.0.0.1:8443"

    monkeypatch.delenv("HTTPS_PROXY")
    assert proxy.resolve_proxy("system").url == "http://127.0.0.1:8080"

    monkeypatch.delenv("HTTP_PROXY")
    assert proxy.resolve_proxy("system").url is None


def test_normalize_windows_proxy_server():
    assert proxy._normalize_proxy("127.0.0.1:7897") == "http://127.0.0.1:7897"
    assert proxy._normalize_proxy("http=127.0.0.1:7897;https=127.0.0.1:7898") == "http://127.0.0.1:7898"


def test_proxy_tcp_failure_uses_user_message(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(proxy.socket, "create_connection", fail)
    result = proxy.proxy_tcp_test("http://127.0.0.1:7897")
    assert result["status"] == "FAIL"
    assert result["message"] == "当前系统代理端口不可用，请检查代理软件。"


def test_proxy_mode_none_never_uses_system_or_environment(monkeypatch):
    monkeypatch.setattr(proxy, "read_windows_proxy", lambda: proxy.ProxyInfo("http://127.0.0.1:7897", "windows-system", True))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8443")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8080")

    resolved = proxy.resolve_proxy("none")

    assert resolved.url is None
    assert resolved.source == "direct"
