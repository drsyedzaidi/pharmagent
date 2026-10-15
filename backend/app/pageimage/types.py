"""Pydantic locator / bounding-box types for the page-image viewer."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pageimage.limits import DEFAULT_SCALE, MAX_PAGES, MAX_SCALE, MIN_SCALE

DOC_ID_PATTERN = r"^[0-9a-f]{32}$"


class PageLocator(BaseModel):
    """Which page of which stored document. ``page`` is 0-based."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(pattern=DOC_ID_PATTERN)
    page: int = Field(ge=0, lt=MAX_PAGES)


class NormBBox(BaseModel):
    """Axis-aligned box in page-normalized coordinates: origin top-left,
    x right, y down, every value in [0, 1], x0 < x1 and y0 < y1."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    x0: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y0: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    x1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered(self) -> NormBBox:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError("bbox must satisfy x0 < x1 and y0 < y1")
        return self


class RenderParams(BaseModel):
    """Validated render options. ``bbox`` (optional) crops the output."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    scale: float = Field(default=DEFAULT_SCALE, ge=MIN_SCALE, le=MAX_SCALE, allow_inf_nan=False)
    bbox: NormBBox | None = None
