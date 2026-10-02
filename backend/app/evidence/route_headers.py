"""Evidence routes keep the headers of the HTTPException they raise.

The application's global HTTPException handler (main.py) builds the error envelope without
``exc.headers``, so a ``Retry-After`` set by an evidence route would never reach the client. This route
class answers such an exception itself, in the same envelope and with the same X-Request-ID, so the
503 of a busy parser carries its Retry-After. Exceptions without headers go to the global handler
unchanged."""
from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute


class HeaderKeepingRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def keep_headers(request: Request) -> Response:
            try:
                return await handler(request)
            except HTTPException as exc:
                if not exc.headers:
                    raise
                rid = getattr(request.state, "request_id", "")
                return JSONResponse(
                    status_code=exc.status_code, headers={**exc.headers, "X-Request-ID": rid},
                    content={"success": False, "error": {"code": f"http_{exc.status_code}",
                                                         "message": str(exc.detail), "request_id": rid}})

        return keep_headers
