from scripts.preflight_expert_consensus_banks import summarize


def test_summarize_activity() -> None:
    assert summarize([0.0, 2.0, 4.0]) == {
        "mean": 2.0,
        "min": 0.0,
        "max": 4.0,
        "nonzero_fraction": 2 / 3,
    }
