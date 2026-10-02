"""The global HTTPException handler keeps ``exc.headers`` (Retry-After, WWW-Authenticate, ...) and the
envelope and X-Request-ID stay the handler's own. Hop-by-hop, cookie, framing and CORS headers are dropped."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.main as main
from app.response_headers import safe_exception_headers

PATH = "/api/_test_exception_headers"


def _mount_first(path: str, fn) -> None:
    """Register *fn* ahead of main's SPA catch-all (present when a frontend build exists)."""
    main.app.add_api_route(path, fn, methods=["GET"])
    main.app.router.routes.insert(0, main.app.router.routes.pop())


def _unmount(path: str) -> None:
    main.app.router.routes[:] = [r for r in main.app.router.routes if getattr(r, "path", "") != path]


@pytest.fixture()
def client():
    def boom(kind: str = "plain"):
        if kind == "plain":
            raise HTTPException(418, "teapot")
        if kind == "custom":
            raise HTTPException(429, "slow down", headers={"Retry-After": "9", "X-Custom": "yes"})
        if kind == "auth":
            raise HTTPException(401, "nope", headers={"WWW-Authenticate": "Bearer"})
        if kind == "override":
            raise HTTPException(400, "bad", headers={
                "X-Request-ID": "forged", "Content-Type": "text/plain", "Content-Length": "1",
                "Set-Cookie": "sid=1", "Connection": "close", "Transfer-Encoding": "chunked",
                "Access-Control-Allow-Origin": "*", "Retry-After": "3"})
        raise AssertionError(kind)

    _mount_first(PATH, boom)
    try:
        yield TestClient(main.app)
    finally:
        _unmount(PATH)


def test_exception_header_reaches_client_with_envelope_and_request_id(client):
    res = client.get(PATH, params={"kind": "custom"}, headers={"X-Request-ID": "rid-1"})
    assert res.status_code == 429
    assert res.headers["Retry-After"] == "9" and res.headers["X-Custom"] == "yes"
    assert res.headers["X-Request-ID"] == "rid-1"
    assert res.headers["content-type"].startswith("application/json")
    assert res.json() == {"success": False, "error": {"code": "http_429", "message": "slow down",
                                                       "request_id": "rid-1"}}


def test_exception_without_headers_is_unchanged(client):
    res = client.get(PATH)
    assert res.status_code == 418
    assert res.json()["error"]["code"] == "http_418" and res.json()["error"]["message"] == "teapot"
    assert res.headers["X-Request-ID"] and "retry-after" not in res.headers
    assert "x-custom" not in res.headers


def test_www_authenticate_on_401_is_forwarded(client):
    res = client.get(PATH, params={"kind": "auth"})
    assert res.status_code == 401 and res.headers["WWW-Authenticate"] == "Bearer"


def test_unsafe_headers_never_override_or_leak(client):
    res = client.get(PATH, params={"kind": "override"}, headers={"X-Request-ID": "rid-2"})
    assert res.status_code == 400
    assert res.headers["X-Request-ID"] == "rid-2"
    assert res.headers["content-type"].startswith("application/json")
    assert res.json()["error"]["request_id"] == "rid-2"          # body intact, so Content-Length was ours
    for name in ("set-cookie", "connection", "transfer-encoding", "access-control-allow-origin"):
        assert name not in res.headers
    assert res.headers["Retry-After"] == "3"                        # the safe one still passes


def test_safe_exception_headers_filters_case_insensitively_and_by_value():
    out = safe_exception_headers({"retry-after": "1", "SET-COOKIE": "a=b", "x-ok": "v", "Keep-Alive": "t",
                                  "X-Bad": "a\r\nSet-Cookie: x=y", "Access-Control-Max-Age": "1"})
    assert out == {"retry-after": "1", "x-ok": "v"}
    assert safe_exception_headers(None) == {} and safe_exception_headers({}) == {}


def test_busy_parser_503_keeps_retry_after_through_the_global_handler(client):
    from app.evidence.parse_queue import ParseBusyError
    from app.evidence.routes import http_error

    def busy():
        raise http_error(ParseBusyError("busy", retry_after=7))

    _mount_first(PATH + "_busy", busy)
    try:
        res = client.get(PATH + "_busy")
    finally:
        _unmount(PATH + "_busy")
    assert res.status_code == 503 and res.headers["Retry-After"] == "7"
