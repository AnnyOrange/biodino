import numpy as np

from scripts.apply_expert_metadata_overlay import build_labeled_payload


def test_build_labeled_payload_preserves_features_and_recomputes_routing() -> None:
    bank = {
        "keys": np.asarray(["a", "b"]),
        "features": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        "reliability": np.asarray([0.2, 0.2], dtype=np.float32),
        "expert_role": np.asarray("tissue"),
    }
    overlay = {
        "keys": np.asarray(["a", "b"]),
        "domain": np.asarray(["d1", "d2"]),
        "organism": np.asarray(["human", "mouse"]),
        "acquisition_family": np.asarray(["histopathology", "fluorescence_microscopy"]),
        "sample_type": np.asarray(["tissue", "cell"]),
    }
    payload = build_labeled_payload(bank, overlay, expert_role="tissue")
    np.testing.assert_array_equal(payload["features"], bank["features"])
    np.testing.assert_allclose(payload["reliability"], [1.0, 0.15])
    assert payload["organism"].tolist() == ["human", "mouse"]


def test_build_labeled_payload_rejects_key_mismatch() -> None:
    bank = {"keys": np.asarray(["a"]), "features": np.asarray([[1.0]])}
    overlay = {
        "keys": np.asarray(["b"]),
        "domain": np.asarray(["d"]),
        "organism": np.asarray(["human"]),
        "acquisition_family": np.asarray(["histopathology"]),
        "sample_type": np.asarray(["tissue"]),
    }
    try:
        build_labeled_payload(bank, overlay, expert_role="tissue")
    except ValueError as error:
        assert "keys differ" in str(error)
    else:
        raise AssertionError("Expected key mismatch to fail")
