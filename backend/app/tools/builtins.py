"""Assemble the default tool registry from all tool modules."""
from __future__ import annotations

from app.tools import (
    be_tools,
    compartmental_tools,
    data_tools,
    dp_tools,
    engine_tools,
    nca_tools,
    pkmodel_tools,
    poppk_tools,
    qc_tools,
    regulatory_tools,
    report_tools,
    review_tools,
    simest_tools,
    viz_tools,
)
from app.tools.base import ToolRegistry


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for mod in (data_tools, nca_tools, be_tools, dp_tools, compartmental_tools,
                poppk_tools, pkmodel_tools, engine_tools, qc_tools, report_tools,
                regulatory_tools, review_tools, viz_tools, simest_tools):
        for tool in mod.TOOLS:
            reg.register(tool)
    return reg
