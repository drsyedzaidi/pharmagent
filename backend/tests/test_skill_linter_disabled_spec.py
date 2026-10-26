"""A disabled specification may name artifacts that are not live PharmState fields."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.skill_linter import check_skills


def _write_skill(root: Path, name: str, frontmatter_extra: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo skill for linter tests.\n{frontmatter_extra}---\n\n# {name}\n\nBody.\n",
        encoding="utf-8",
    )


def _artifact_finding(root: Path):
    return next(f for f in check_skills(root) if f.check == "produced_artifacts_valid")


def test_live_skill_with_unknown_artifact_name_still_fails(tmp_path: Path):
    _write_skill(tmp_path, "live-skill", "produced_artifacts:\n  - not_a_state_field\n")
    f = _artifact_finding(tmp_path)
    assert f.status == "fail"
    assert "not_a_state_field" in f.detail


def test_explicitly_runtime_enabled_skill_with_unknown_artifact_still_fails(tmp_path: Path):
    _write_skill(tmp_path, "enabled-skill", "runtime_available: true\nproduced_artifacts:\n  - not_a_state_field\n")
    assert _artifact_finding(tmp_path).status == "fail"


def test_disabled_specification_may_name_future_artifacts(tmp_path: Path):
    _write_skill(tmp_path, "disabled-spec", "runtime_available: false\nproduced_artifacts:\n  - future_records_spec\n")
    assert _artifact_finding(tmp_path).status == "pass"


@pytest.mark.parametrize("flag", ["'false'", "0", "no_such"])
def test_only_a_real_false_disables_the_artifact_check(tmp_path: Path, flag: str):
    _write_skill(tmp_path, "odd-flag", f"runtime_available: {flag}\nproduced_artifacts:\n  - not_a_state_field\n")
    assert _artifact_finding(tmp_path).status == "fail"


def test_disabled_specification_still_requires_a_list_of_strings(tmp_path: Path):
    _write_skill(tmp_path, "bad-shape", "runtime_available: false\nproduced_artifacts: not-a-list\n")
    assert _artifact_finding(tmp_path).status == "fail"
