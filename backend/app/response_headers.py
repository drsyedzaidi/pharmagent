"""Which headers of a raised ``HTTPException`` may reach the client.

The global HTTPException handler forwards ``exc.headers`` (``Retry-After`` on a 429/503,
``WWW-Authenticate`` on a 401, ``Allow`` on a 405). It builds the envelope, ``Content-Type``,
``Content-Length`` and ``X-Request-ID`` itself, so nothing raised by a route may replace them. A deny-list
rather than an allow-list: routes may add their own informational headers, and the denied set is the part
that is unsafe or wrong on a response the handler builds:

* hop-by-hop headers (RFC 9110 section 7.6.1) and the framing headers ``Content-Length`` / ``Content-Type`` /
  ``Content-Encoding`` / ``Transfer-Encoding``, which would corrupt the body the handler writes;
* ``Set-Cookie`` / ``Set-Cookie2``: an error path must not plant or clear cookies;
* ``Access-Control-*``: CORS policy belongs to the middleware alone;
* ``X-Request-ID``: owned by the request-context middleware.

A name or value containing control characters is dropped too (header splitting)."""
from __future__ import annotations

from collections.abc import Mapping

DENIED = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
    "trailer", "trailers", "transfer-encoding", "upgrade",
    "content-length", "content-type", "content-encoding",
    "set-cookie", "set-cookie2", "x-request-id",
})
DENIED_PREFIXES = ("access-control-",)


def _clean(text: str) -> bool:
    return all(c == "\t" or (" " <= c and c != "\x7f") for c in text)


def safe_exception_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """The headers of an HTTPException that may be sent (names keep the caller's spelling)."""
    out: dict[str, str] = {}
    for name, value in (headers or {}).items():
        low = name.lower()
        if not name or low in DENIED or low.startswith(DENIED_PREFIXES):
            continue
        if not (_clean(name) and _clean(str(value))):
            continue
        out[name] = str(value)
    return out
