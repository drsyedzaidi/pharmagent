"""Security review LOW 6 and LOW 7."""
from __future__ import annotations


def test_startup_validation_fills_the_shared_catalog_cache():
    from app.conventions.catalog_io import cached_catalog
    from app.conventions.runtime import validate_catalog_at_startup

    cached_catalog.cache_clear()
    version = validate_catalog_at_startup()
    assert cached_catalog.cache_info().currsize == 1                 # the validated load IS the fingerprinted one
    assert version == cached_catalog().catalog.catalog_version


def test_generate_272_keeps_its_docstring():
    from app.tools.regulatory_tools import generate_272

    assert generate_272.__doc__ and generate_272.__doc__.strip()
