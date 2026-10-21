"""Conventions catalog: the analysis conventions PharmAgent applies, as reviewable, test-linked data.

Slice 1 is read-only. Nothing in the application imports this package; it changes no runtime behaviour.
"""
from .catalog_io import DEFAULT_CATALOG, load_catalog
from .coverage import CoverageProblem, check_test_coverage
from .render import render_report_json, render_report_markdown, rule_text
from .schema import Approval, BranchPoint, Catalog, Convention, Source

__all__ = ["Approval", "BranchPoint", "Catalog", "Convention", "CoverageProblem", "DEFAULT_CATALOG", "Source",
           "check_test_coverage", "load_catalog", "render_report_json", "render_report_markdown", "rule_text"]
