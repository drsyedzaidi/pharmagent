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

A name that is not an RFC 9110 token (a space, a colon, a control character: ``"X-Request-Id "`` is not
``X-Request-ID`` to the deny-list but is to some peers) is dropped, and so is a value with a control character
(header splitting) or a character outside latin-1 (it cannot be put on the wire: dropping it keeps the
original status instead of turning the error into a 500)."""
from __future__ import annotations

import re
from collections.abc import Mapping

DENIED = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
    "trailer", "trailers", "transfer-encoding", "upgrade",
    "content-length", "content-type", "content-encoding",
    "set-cookie", "set-cookie2", "x-request-id",
})
DENIED_PREFIXES = ("access-control-",)


_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def _valid_name(name: str) -> bool:
    return _TOKEN.fullmatch(name) is not None


def _valid_value(value: str) -> bool:
    if not all(c == "\t" or (" " <= c and c != "\x7f") for c in value):
        return False
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def safe_exception_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """The headers of an HTTPException that may be sent (names keep the caller's spelling)."""
    out: dict[str, str] = {}
    for name, value in (headers or {}).items():
        low = name.lower()
        if not _valid_name(name) or low in DENIED or low.startswith(DENIED_PREFIXES):
            continue
        if not _valid_value(str(value)):
            continue
        out[name] = str(value)
    return out
