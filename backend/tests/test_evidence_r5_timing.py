"""evidence_timing.limit(): strict without a tracer, scaled while one is installed (and only then)."""
from __future__ import annotations

import sys
import threading

import pytest

from tests import evidence_timing as ET


@pytest.fixture
def untraced(monkeypatch):
    monkeypatch.setattr(sys, "gettrace", lambda: None)
    monkeypatch.setattr(threading, "gettrace", lambda: None)
    if hasattr(sys, "monitoring"):
        monkeypatch.setattr(sys.monitoring, "get_tool", lambda _id: None)


def test_limit_is_the_strict_value_when_nothing_is_tracing(untraced):
    assert ET.tracing_active() is False
    assert ET.limit(4.0) == 4.0


def test_limit_scales_when_a_main_thread_tracer_is_installed(untraced, monkeypatch):
    monkeypatch.setattr(sys, "gettrace", lambda: (lambda *a: None))
    assert ET.tracing_active() is True
    assert ET.limit(4.0) == 4.0 * ET.TRACE_SLOWDOWN


def test_limit_scales_when_only_new_threads_are_traced(untraced, monkeypatch):
    monkeypatch.setattr(threading, "gettrace", lambda: (lambda *a: None))
    assert ET.limit(2.0) == 2.0 * ET.TRACE_SLOWDOWN


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="sys.monitoring needs Python 3.12+")
def test_limit_scales_when_a_monitoring_tool_is_registered(untraced, monkeypatch):
    monkeypatch.setattr(sys.monitoring, "get_tool", lambda tool_id: "coverage" if tool_id == 1 else None)
    assert ET.limit(1.0) == ET.TRACE_SLOWDOWN


def test_the_scaled_limit_still_rejects_an_order_of_magnitude_regression():
    assert 4.0 * ET.TRACE_SLOWDOWN < 4.0 * 10
