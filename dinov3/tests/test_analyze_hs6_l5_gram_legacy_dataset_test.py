from pathlib import Path

import pytest

from scripts.analyze_hs6_l5_gram_legacy_dataset_test import (
    ENDPOINT_CHECKPOINT_ID,
    EXPECTED_LANES,
    infer_extra_arm_state_root,
    lane_audit_from_state,
)


def _state_fixture(root: Path, *, missing: str | None = None, terminal: str | None = None) -> Path:
    state = root / "_state"
    done = state / "done"
    done.mkdir(parents=True)
    for lane in EXPECTED_LANES - ({missing} if missing else set()):
        (done / f"ckpt_{ENDPOINT_CHECKPOINT_ID}__{lane}.json").write_text("{}\n")
    if terminal:
        terminal_root = state / "terminal"
        terminal_root.mkdir()
        (terminal_root / f"ckpt_{ENDPOINT_CHECKPOINT_ID}__{terminal}.json").write_text("{}\n")
    return state


def test_extra_arm_state_audit_requires_exactly_all_twelve_lanes(tmp_path: Path) -> None:
    point = tmp_path / "arm" / f"point_{ENDPOINT_CHECKPOINT_ID}"
    point.mkdir(parents=True)
    _state_fixture(point.parent)

    audit = lane_audit_from_state(infer_extra_arm_state_root(point))

    assert audit["complete"] is True
    assert audit["done_lanes"] == sorted(EXPECTED_LANES)
    assert audit["missing_lanes"] == []
    assert audit["unexpected_done_lanes"] == []
    assert audit["terminal"] == []


def test_online_done_cannot_replace_a_missing_lane(tmp_path: Path) -> None:
    point = tmp_path / "arm" / f"point_{ENDPOINT_CHECKPOINT_ID}"
    point.mkdir(parents=True)
    state = _state_fixture(point.parent, missing="ood")
    online = point.parent / "_online_status"
    online.mkdir()
    (online / f"ckpt_{ENDPOINT_CHECKPOINT_ID}.done").write_text("{}\n")

    audit = lane_audit_from_state(state)

    assert audit["complete"] is False
    assert audit["missing_lanes"] == ["ood"]


def test_terminal_marker_fails_a_complete_lane_set(tmp_path: Path) -> None:
    state = _state_fixture(tmp_path / "arm", terminal="retrieval")

    audit = lane_audit_from_state(state)

    assert audit["done_lanes"] == sorted(EXPECTED_LANES)
    assert audit["complete"] is False
    assert audit["terminal"] == [f"ckpt_{ENDPOINT_CHECKPOINT_ID}__retrieval.json"]


def test_nonstandard_point_requires_explicit_existing_state(tmp_path: Path) -> None:
    point = tmp_path / "arm" / "endpoint"
    point.mkdir(parents=True)

    with pytest.raises(ValueError, match="cannot infer extra-arm state"):
        infer_extra_arm_state_root(point)

    state = _state_fixture(tmp_path / "separate")
    assert infer_extra_arm_state_root(point, state) == state.resolve()
