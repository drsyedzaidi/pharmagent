"""Suite-wide isolation for the mikebrain rule catalog.

``adversarial_review`` reads the repository's rule catalog. Once the owner approves a real rule, any test
that runs a review on a state that happens to satisfy it would start producing an extra finding. So every
test sees an EMPTY catalog unless it points ``catalog_io.DEFAULT_CATALOG`` somewhere itself
(``test_knowledge_rules.py`` does, and reads the shipped file through ``SHIPPED_CATALOG``).
"""
from __future__ import annotations

import json

import pytest

from app.knowledge.rules import catalog_io


@pytest.fixture(autouse=True)
def _empty_rule_catalog(tmp_path_factory, monkeypatch):
    path = tmp_path_factory.mktemp("rule_catalog") / "catalog.json"
    path.write_text(json.dumps({"catalog_version": "0.0.0", "rules": []}), encoding="utf-8")
    monkeypatch.setattr(catalog_io, "DEFAULT_CATALOG", path)
    catalog_io.cached_rule_catalog.cache_clear()
    yield
    catalog_io.cached_rule_catalog.cache_clear()
