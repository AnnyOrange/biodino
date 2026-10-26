"""Pin an isolated remote FM overlay without changing the main-cluster adapter."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "outputs/02_eval_runs/unified_paper/remote_fm_lc25000_stage"
CANONICAL = Path("/mnt/huawei_deepcad/benchmark_model")
PIN = ROOT / "dinov3/eval/bio_frozen_eval/external_fm_source_hashes.json"
ADAPTED = {
    "benchmark_eval/encoders.py",
    "benchmark_eval/rules_features.py",
    "run_fm_rules.py",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    original = json.loads(PIN.read_text())
    copied = {"protocol_id": original["protocol_id"], "files": {}}
    proof = {"canonical_root": str(CANONICAL), "remote_root_env": "UNIFIED_PAPER_BENCHMARK_MODEL_ROOT", "files": {}}
    for name, expected in original["files"].items():
        relative = str(Path(name).relative_to(CANONICAL))
        actual_local = sha(CANONICAL / relative)
        if actual_local != expected:
            raise ValueError(f"Pinned local FM source changed: {relative}")
        staged = sha(STAGE / "benchmark_model" / relative)
        if relative not in ADAPTED and staged != expected:
            raise ValueError(f"Unexpected remote source change: {relative}")
        if relative in ADAPTED and staged == expected:
            raise ValueError(f"Missing root adaptation: {relative}")
        copied["files"][name] = staged
        proof["files"][relative] = {"main_sha256": expected, "overlay_sha256": staged, "path_only_adaptation": relative in ADAPTED}
    save(STAGE / "dinov3/eval/bio_frozen_eval/external_fm_source_hashes.json", copied)
    save(STAGE / "outputs/02_eval_runs/unified_paper/validation/remote_fm_source_overlay.json", proof)
    print("PINNED", len(proof["files"]), "sources; path-only overlays", len(ADAPTED))


if __name__ == "__main__":
    main()
