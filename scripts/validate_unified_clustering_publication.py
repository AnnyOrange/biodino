"""Check that the canonical paper tables use the corrected clustering statistics."""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/02_eval_runs/unified_paper"
REPORT = ROOT / "outputs/00_reports/hs0_hs6_1tb_5tb_fm14_20260924"
VERSION = "clustering-paired-source-group-centered-normal-bootstrap-v2"


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10)


def main():
    paper = [r for r in rows(REPORT / "paper_results.csv") if r["task"] == "clustering"]
    indexed = [r for r in paper if r["statistics_path"] and r["ci_low"] and r["ci_high"]]
    assert indexed, "No clustering confidence intervals in paper results"
    for row in indexed:
        path = Path(row["statistics_path"])
        assert path.is_file(), path
        stat = json.loads(path.read_text())
        assert stat["provenance"]["implementation"] == VERSION, path
        nmi = stat["metrics"]["nmi"]
        assert stat["model"] == row["model"] and stat["dataset"] == row["dataset"]
        assert all(close(row[public], nmi[source]) for public, source in
                   [("value", "value"), ("ci_low", "ci_low"), ("ci_high", "ci_high")]), path
        assert float(row["ci_low"]) <= float(row["value"]) <= float(row["ci_high"]), path

    pairs = rows(REPORT / "clustering_paired_statistics.csv")
    assert pairs, "No clustering comparisons in paper report"
    for row in pairs:
        path = Path(row["source"])
        assert path.is_file(), path
        stat = json.loads(path.read_text())
        assert stat["provenance"]["implementation"] == VERSION, path
        nmi = stat["metrics"]["nmi"]
        assert all(close(row[public], nmi[source]) for public, source in
                   [("difference", "difference"), ("ci_low", "ci_low"),
                    ("ci_high", "ci_high"), ("p_two_sided", "p_two_sided")]), path
        assert float(row["ci_low"]) <= float(row["difference"]) <= float(row["ci_high"]), path
        assert 0 <= float(row["p_holm"]) <= 1, path

    history = OUT / "statistics/history/global"
    archived = sum(1 for p in history.glob("*.json") if p.is_file())
    assert archived, "Old statistics were not archived"
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "implementation": VERSION,
        "paper_clustering_rows_with_matching_ci": len(indexed),
        "paper_clustering_rows_without_ci": len(paper) - len(indexed),
        "paired_comparisons_matching_source_and_holm": len(pairs),
        "archived_statistics_files": archived,
        "paper_csv": str(REPORT / "paper_results.csv"),
        "paired_csv": str(REPORT / "clustering_paired_statistics.csv"),
        "method_config": str(OUT / "clustering_statistics_config.json"),
    }
    target = OUT / "validation/clustering_ci_v2_publication.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
