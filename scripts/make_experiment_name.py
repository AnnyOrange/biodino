#!/usr/bin/env python3
"""Generate and validate canonical experiment ids."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


KIND_TO_DIR = {
    "train": "outputs/01_training_runs",
    "eval": "outputs/02_eval_runs",
    "cmp": "outputs/03_comparisons",
    "debug": "outputs/05_debug_smoke",
    "smoke": "outputs/05_debug_smoke",
    "data": "outputs/06_data_prep_transfer",
    "report": "outputs/00_reports",
}

FIELD_RE = r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
NAME_RE = re.compile(
    rf"^\d{{8}}__(?:{'|'.join(KIND_TO_DIR)})__{FIELD_RE}__{FIELD_RE}__{FIELD_RE}(?:__{FIELD_RE})?$"
)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        raise ValueError("empty field after slugification")
    return value


def build_id(args: argparse.Namespace) -> str:
    date = args.date or dt.datetime.now().strftime("%Y%m%d")
    fields = [date, args.kind, args.model, args.data, args.protocol]
    if args.tag:
        fields.append(args.tag)
    return "__".join([fields[0], fields[1], *(slugify(x) for x in fields[2:])])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or validate canonical experiment ids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("kind", nargs="?", choices=sorted(KIND_TO_DIR))
    parser.add_argument("model", nargs="?")
    parser.add_argument("data", nargs="?")
    parser.add_argument("protocol", nargs="?")
    parser.add_argument("--tag", default="")
    parser.add_argument("--date", default="", help="YYYYMMDD; defaults to today")
    parser.add_argument("--path", action="store_true", help="print the category path instead of only the id")
    parser.add_argument("--id-only", action="store_true", help="print only the id even if --path is set")
    parser.add_argument("--validate", metavar="EXPERIMENT_ID", help="validate an existing id and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate:
        if NAME_RE.match(args.validate):
            print(f"valid: {args.validate}")
            return 0
        print(f"invalid: {args.validate}")
        print("expected: YYYYMMDD__<kind>__<model>__<data>__<protocol>[__<tag>]")
        return 2

    missing = [name for name in ("kind", "model", "data", "protocol") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing required fields: {', '.join(missing)}")

    exp_id = build_id(args)
    if not NAME_RE.match(exp_id):
        raise SystemExit(f"generated invalid id: {exp_id}")
    if args.path and not args.id_only:
        print(Path(KIND_TO_DIR[args.kind]) / exp_id)
    else:
        print(exp_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
