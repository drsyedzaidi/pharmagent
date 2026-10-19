"""The UNSAFE operator override excuses ONE thing: a missing RLIMIT_AS, on a platform that cannot enforce it.

CPU time, file size, core size and open-file limits must stay enforced whatever the override says, and on Linux, where
RLIMIT_AS exists, it must stay enforced too. The checks live in two places that must agree: the worker (decides whether to
run at all) and the parent (validates the attestation the worker sends back)."""
from __future__ import annotations

import json

import pytest

from app.config import settings
from app.pageimage import runner, worker
from app.pageimage.runner import RenderError
from tests.pdf_fixtures import valid_report

NON_AS = ["RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_CORE", "RLIMIT_NOFILE"]


def _stderr(**unenforced: bool) -> bytes:
    rep = valid_report()
    for e in rep["pageimage_limits"]["limits"]:
        if e["name"] in unenforced:
            e.update(state="unenforced", soft=None, hard=None)
    return (json.dumps(rep) + "\n").encode()


@pytest.fixture
def override_on(monkeypatch):
    monkeypatch.setattr(settings, "pageimage_unsafe_allow_unenforced_limits", True)


@pytest.fixture
def no_rlimit_as(monkeypatch):
    """A platform (macOS) on which RLIMIT_AS cannot be enforced."""
    monkeypatch.setattr(runner, "_as_is_enforceable", lambda: False)
    monkeypatch.setattr(worker, "_as_is_enforceable", lambda: False)


@pytest.fixture
def linux_like(monkeypatch):
    monkeypatch.setattr(runner, "_as_is_enforceable", lambda: True)
    monkeypatch.setattr(worker, "_as_is_enforceable", lambda: True)


# ───────────────────────── parent: validating the attestation ─────────────────────────

def test_the_override_excuses_a_missing_rlimit_as_where_the_platform_has_none(override_on, no_rlimit_as):
    assert runner._attest(_stderr(RLIMIT_AS=True)) is False                  # accepted, reported as not fully enforced


@pytest.mark.parametrize("name", NON_AS)
def test_the_override_never_excuses_any_other_limit(override_on, no_rlimit_as, name):
    with pytest.raises(RenderError) as e:
        runner._attest(_stderr(**{name: True}))
    assert e.value.kind == "limit_unavailable"


@pytest.mark.parametrize("name", NON_AS)
def test_one_excusable_limit_does_not_launder_another(override_on, no_rlimit_as, name):
    with pytest.raises(RenderError):
        runner._attest(_stderr(RLIMIT_AS=True, **{name: True}))


def test_on_linux_a_missing_rlimit_as_is_refused_even_with_the_override(override_on, linux_like):
    with pytest.raises(RenderError) as e:
        runner._attest(_stderr(RLIMIT_AS=True))
    assert e.value.kind == "limit_unavailable"


def test_without_the_override_nothing_unenforced_is_accepted(monkeypatch, no_rlimit_as):
    monkeypatch.setattr(settings, "pageimage_unsafe_allow_unenforced_limits", False)
    with pytest.raises(RenderError):
        runner._attest(_stderr(RLIMIT_AS=True))


def test_a_fully_enforced_report_is_accepted_everywhere(override_on, linux_like):
    assert runner._attest(_stderr()) is True


# ───────────────────────── worker: deciding whether to run ─────────────────────────

class FakeResource:
    """Just enough of ``resource`` to drive apply_limits without touching this process's real limits."""
    RLIM_INFINITY = -1
    RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_CORE, RLIMIT_NOFILE = range(5)

    def __init__(self, refuse=(), unconfirmed=()):
        self.refuse, self.unconfirmed = set(refuse), set(unconfirmed)
        self.limits = {i: (-1, -1) for i in range(5)}

    def getrlimit(self, which):
        return self.limits[which]

    def setrlimit(self, which, value):
        if which in self.refuse:
            raise OSError("refused")
        if which not in self.unconfirmed:
            self.limits[which] = value


def _apply(monkeypatch, fake, allow):
    monkeypatch.setitem(__import__("sys").modules, "resource", fake)
    return worker.apply_limits(10, 1 << 30, allow)


@pytest.mark.parametrize("name", NON_AS)
@pytest.mark.parametrize("how", ["refuse", "unconfirmed"])
def test_the_worker_refuses_to_run_without_any_non_as_limit_even_with_the_override(monkeypatch, no_rlimit_as, name, how):
    which = getattr(FakeResource, name)
    fake = FakeResource(**{how: [which]})
    with pytest.raises(worker.LimitUnavailable):
        _apply(monkeypatch, fake, True)


def test_the_worker_runs_without_rlimit_as_only_where_the_platform_lacks_it_and_the_override_is_set(monkeypatch,
                                                                                                    no_rlimit_as):
    rep = _apply(monkeypatch, FakeResource(refuse=[FakeResource.RLIMIT_AS]), True)
    states = {r["name"]: r["state"] for r in rep}
    assert states.pop("RLIMIT_AS") == "unenforced" and set(states.values()) == {"enforced"}
    with pytest.raises(worker.LimitUnavailable):
        _apply(monkeypatch, FakeResource(refuse=[FakeResource.RLIMIT_AS]), False)      # no override: refuse


def test_the_worker_refuses_a_missing_rlimit_as_on_linux_even_with_the_override(monkeypatch, linux_like):
    with pytest.raises(worker.LimitUnavailable):
        _apply(monkeypatch, FakeResource(refuse=[FakeResource.RLIMIT_AS]), True)


def test_the_worker_and_the_parent_agree_on_the_platform_rule():
    assert worker._as_is_enforceable() == runner._as_is_enforceable()
