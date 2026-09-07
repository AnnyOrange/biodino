#!/usr/bin/env python3
"""Enrich packed source-domain counts with official IDR study metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_GITMODULES = "https://raw.githubusercontent.com/IDR/idr-metadata/master/.gitmodules"
ACCESSION = re.compile(r"^(idr\d{4})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gitmodules-url", default=DEFAULT_GITMODULES)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def fetch_text(url: str, *, timeout: float, attempts: int = 3) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "BioDINO-domain-audit/1.0"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8-sig", errors="replace")
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(1.0 + attempt)
    raise RuntimeError("unreachable")


def parse_submodule_repositories(text: str) -> dict[str, str]:
    repositories = {}
    current_path = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("path = "):
            current_path = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("url = ") and current_path:
            match = ACCESSION.match(current_path)
            if match:
                repositories[match.group(1).lower()] = current_path
            current_path = None
    return repositories


def parse_study_fields(text: str) -> dict[str, list[str]]:
    fields = defaultdict(list)
    for raw_line in text.splitlines():
        line = raw_line.strip("\r\n")
        if not line or line.lstrip().startswith("#"):
            continue
        cells = [cell.strip() for cell in line.split("\t")]
        key = cells[0]
        for value in cells[1:]:
            if value and value not in fields[key]:
                fields[key].append(value)
    return dict(fields)


def acquisition_family(imaging_methods: str, study_type: str, keywords: str) -> str:
    text = " ".join((imaging_methods, study_type, keywords)).lower()
    if any(token in text for token in ("light sheet", "lightsheet", "spim", "mesospim")):
        return "light_sheet_fluorescence"
    if any(token in text for token in ("histopath", "whole slide", "h&e", "brightfield histology")):
        return "histopathology"
    if any(token in text for token in ("electron microscopy", "electron micrograph", "tem", "sem")):
        return "electron_microscopy"
    if any(token in text for token in ("mass cytometry", "imaging mass", "hyperion")):
        return "imaging_mass_cytometry"
    if any(token in text for token in ("phase contrast", "dic", "brightfield", "transmitted light")):
        return "label_free_light_microscopy"
    if any(
        token in text
        for token in (
            "fluorescence",
            "confocal",
            "widefield",
            "wide-field",
            "tirf",
            "spinning disk",
            "super-resolution",
            "high content screening",
        )
    ):
        return "fluorescence_microscopy"
    if any(token in text for token in ("optical coherence tomography", "x-ray", "computed tomography")):
        return "medical_optical_or_xray"
    return "unresolved"


def joined(fields: dict[str, list[str]], key: str) -> str:
    return " | ".join(fields.get(key, ()))


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    gitmodules = fetch_text(args.gitmodules_url, timeout=args.timeout)
    repositories = parse_submodule_repositories(gitmodules)

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    enriched = []
    for row in rows:
        domain = row["domain"]
        if not domain.startswith("idr:"):
            enriched.append(
                {
                    **row,
                    "metadata_status": "not_idr",
                    "study_title": "",
                    "study_type": "",
                    "organism": "",
                    "imaging_method": "",
                    "sample_type": "",
                    "acquisition_family": "unresolved",
                    "study_keywords": "",
                    "metadata_url": "",
                }
            )
            continue

        match = ACCESSION.match(domain.split(":", 1)[1])
        accession = match.group(1).lower() if match else ""
        repository = repositories.get(accession)
        if not repository:
            status = "missing_repository"
            fields = {}
            url = ""
        else:
            url = (
                f"https://raw.githubusercontent.com/IDR/{repository}/HEAD/"
                f"{accession}-study.txt"
            )
            cache_path = args.cache_dir / f"{accession}-study.txt"
            try:
                if cache_path.exists():
                    study_text = cache_path.read_text(encoding="utf-8", errors="replace")
                else:
                    study_text = fetch_text(url, timeout=args.timeout)
                    cache_path.write_text(study_text, encoding="utf-8")
                fields = parse_study_fields(study_text)
                status = "ok"
            except (urllib.error.URLError, TimeoutError):
                fields = {}
                status = "fetch_failed"

        imaging_method = joined(fields, "Experiment Imaging Method")
        study_type = joined(fields, "Study Type")
        keywords = joined(fields, "Study Key Words")
        enriched.append(
            {
                **row,
                "metadata_status": status,
                "study_title": joined(fields, "Study Title"),
                "study_type": study_type,
                "organism": joined(fields, "Study Organism"),
                "imaging_method": imaging_method,
                "sample_type": joined(fields, "Experiment Sample Type"),
                "acquisition_family": acquisition_family(imaging_method, study_type, keywords),
                "study_keywords": keywords,
                "metadata_url": url,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=enriched[0].keys())
        writer.writeheader()
        writer.writerows(enriched)

    status_rows = Counter()
    acquisition_rows = Counter()
    acquisition_domains = Counter()
    organism_rows = Counter()
    for row in enriched:
        count = int(row["mapping_rows"])
        status_rows[row["metadata_status"]] += count
        acquisition_rows[row["acquisition_family"]] += count
        acquisition_domains[row["acquisition_family"]] += 1
        for organism in (value.strip() for value in row["organism"].split("|") if value.strip()):
            organism_rows[organism] += count
    summary = {
        "source_domains": len(enriched),
        "mapping_rows_by_metadata_status": dict(status_rows.most_common()),
        "mapping_rows_by_acquisition_family": dict(acquisition_rows.most_common()),
        "domains_by_acquisition_family": dict(acquisition_domains.most_common()),
        "top_organisms_by_mapping_rows": dict(organism_rows.most_common(30)),
        "warning": (
            "Acquisition families are deterministic keyword groupings over official IDR fields; "
            "external collections and ambiguous IDR studies still require curation."
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
