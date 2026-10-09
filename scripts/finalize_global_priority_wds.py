#!/usr/bin/env python3
"""Keep exactly the first N successfully decoded global random candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path


def read_status(root: Path, expected: int, repair_roots: list[Path]) -> tuple[list[int], Counter, dict]:
    outcomes = {}
    for path in sorted(root.glob("status_w*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                priority = int(row["priority"])
                if priority in outcomes:
                    raise ValueError(f"Repeated candidate priority: {priority}")
                outcomes[priority] = (row["status"], row["reason"], row.get("source_root") or "unknown")
    if set(outcomes) != set(range(expected)):
        missing = set(range(expected)) - outcomes.keys()
        raise ValueError(f"Status coverage mismatch: {len(outcomes)}/{expected}, missing={len(missing)}")
    for repair_root in repair_roots:
        repaired = set()
        for path in sorted(repair_root.glob("status_w*.csv")):
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    priority = int(row["priority"])
                    if priority in repaired or priority not in outcomes:
                        raise ValueError(f"Repeated or unknown repair priority: {priority}")
                    repaired.add(priority)
                    if row["status"] == "success":
                        if outcomes[priority][0] == "success":
                            raise ValueError(f"Repair duplicates successful candidate: {priority}")
                        outcomes[priority] = ("success", "", outcomes[priority][2])
    reasons = Counter(reason.split(":", 1)[0] for status, reason, _root in outcomes.values()
                      if status != "success")
    by_root = {}
    for status, _reason, root in outcomes.values():
        by_root.setdefault(root, Counter())[status] += 1
    return sorted(p for p, (status, _reason, _root) in outcomes.items()
                  if status == "success"), reasons, by_root


class Output:
    def __init__(self, root: Path, per_shard: int):
        self.root = root
        self.per_shard = per_shard
        self.count = 0
        self.shard = -1
        self.tar = None

    def start_sample(self):
        if self.count % self.per_shard == 0:
            self.close()
            self.shard += 1
            self.tar = tarfile.open(self.root / f"filtered_mixed_train{self.shard:06d}.tar", "w")

    def add_member(self, member: tarfile.TarInfo, payload: bytes):
        assert self.tar is not None
        info = tarfile.TarInfo(member.name)
        info.size = len(payload)
        self.tar.addfile(info, io.BytesIO(payload))

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-root", type=Path, required=True)
    p.add_argument("--repair-root", type=Path, action="append", default=[],
                   help="A priority-preserving repair stage; may be repeated")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--expected-candidates", type=int, required=True)
    p.add_argument("--samples", type=int, default=1_000_000)
    p.add_argument("--samples-per-shard", type=int, default=2000)
    p.add_argument("--max-failure-fraction", type=float, default=0.01,
                   help="Abort rather than silently replace a large inaccessible share")
    p.add_argument("--sampling-population", default="decoded candidates in the supplied manifest",
                   help="The population represented after any source exclusions or missing files")
    a = p.parse_args()
    if a.out_root.exists() and any(a.out_root.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {a.out_root}")
    successes, failed_reasons, status_by_source_root = read_status(
        a.stage_root, a.expected_candidates, a.repair_root)
    failure_fraction = 1.0 - len(successes) / a.expected_candidates
    if failure_fraction > a.max_failure_fraction:
        raise ValueError(f"Candidate failure fraction {failure_fraction:.3%} exceeds "
                         f"{a.max_failure_fraction:.3%}; inspect source access before finalizing")
    if len(successes) < a.samples:
        raise ValueError(f"Only {len(successes)} decoded candidates, need {a.samples}")
    chosen = set(successes[:a.samples])
    cutoff = successes[a.samples - 1]
    a.out_root.mkdir(parents=True, exist_ok=True)
    writer = Output(a.out_root, a.samples_per_shard)
    seen = set()
    sources = Counter()
    channels = Counter()
    digest = hashlib.sha256()
    identities = {}
    reads = Counter()
    source_dtypes = Counter()
    conversions = Counter()
    current_key = None
    current_members = []

    def flush_sample():
        nonlocal current_key, current_members
        if not current_members:
            return
        priority = int(current_key[1:])
        if priority in chosen:
            if priority in seen:
                raise ValueError(f"Repeated sample {priority}")
            metas = [x for x in current_members if x[0].name.endswith(".meta.json")]
            planes = [x for x in current_members if ".ch" in x[0].name and x[0].name.endswith(".tif")]
            if len(metas) != 1 or not planes:
                raise ValueError(f"Incomplete staged sample {current_key}")
            meta = json.loads(metas[0][1])
            if int(meta["priority"]) != priority:
                raise ValueError(f"Metadata priority mismatch {current_key}")
            writer.start_sample()
            for member, payload in current_members:
                writer.add_member(member, payload)
            writer.count += 1
            seen.add(priority)
            identities[priority] = f"{priority}:{meta.get('storage_item_id', meta.get('source_id'))}:{meta.get('frame_idx')}\n"
            sources[str(meta["source_table_code"])] += 1
            channels[len(planes)] += 1
            reads[str(meta.get("read_from", "unspecified"))] += 1
            source_dtypes[str(meta.get("source_dtype", "unspecified"))] += 1
            conversions[str(meta.get("conversion", "unspecified"))] += 1
        current_key, current_members = None, []

    stage_files = sorted(a.stage_root.glob("staged_w*.tar"))
    for repair_root in a.repair_root:
        stage_files.extend(sorted(repair_root.glob("staged_w*.tar")))
    for stage in stage_files:
        with tarfile.open(stage, "r:") as source:
            for member in source:
                if not member.isfile():
                    continue
                key = member.name.split(".", 1)[0]
                if current_key is not None and key != current_key:
                    flush_sample()
                current_key = key
                if int(key[1:]) in chosen:
                    stream = source.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Unreadable staged tar member {stage}:{member.name}")
                    current_members.append((member, stream.read()))
            flush_sample()
    writer.close()
    if seen != chosen or writer.count != a.samples:
        raise ValueError(f"Final sample coverage mismatch: {len(seen)}/{len(chosen)}")
    for priority in sorted(chosen):
        digest.update(identities[priority].encode())
    report = {"samples": writer.count, "candidate_count": a.expected_candidates,
              "sampling_population": a.sampling_population,
              "successful_candidates": len(successes), "last_selected_priority": cutoff,
              "candidate_failure_fraction": failure_fraction,
              "max_selected_priority": max(chosen), "selected_priority_sha256": digest.hexdigest(),
              "failed_reasons": failed_reasons,
              "candidate_status_by_source_root": status_by_source_root,
              "source_table_counts": sources,
              "channel_counts": channels, "read_from_counts": reads,
              "source_dtype_counts": source_dtypes, "conversion_counts": conversions,
              "wds_shards": writer.shard + 1,
              "stage_root": str(a.stage_root),
              "repair_roots": [str(root) for root in a.repair_root],
              "out_root": str(a.out_root)}
    (a.out_root / "finalization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
