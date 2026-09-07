#!/usr/bin/env python3
"""Screen 6-to-3 input mappings and DINOv3 readouts on RxRx3-core.

The RxRx3 parquet files store one JP2 image per channel.  This script groups
six consecutive rows into one well, learns an optional label-free 1x1
``6 -> 3`` convolution on calibration genes, and reports plate-disjoint
same-gene retrieval on held-out genes.  It intentionally lives outside the
main benchmark harness until its protocol and useful settings are validated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dinov3.eval.bio_frozen_eval.encoder import _load_multichannel_stats
from dinov3.eval.bio_segmentation.model_utils import load_dinov3_backbone


DEFAULT_DATA_ROOT = Path("/mnt/huawei_deepcad/benchmark/external_benchmarks_20260901/RxRx3_core")
DEFAULT_OUTPUT = Path("outputs/02_eval_runs/rxrx3_input_layer_screen")


@dataclass(frozen=True)
class WellRecord:
    key: str
    well_id: str
    gene: str
    plate: int
    shard: int
    row: int


def stable_bucket(value: str, modulo: int) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16) % modulo


def source_key_to_well_id(key: str) -> str:
    """Convert ``compound-001/Plate10/AA12_s1_1`` to metadata's well id."""
    experiment, plate, stem = key.split("/")
    address = stem.rsplit("_", 2)[0]
    return f"{experiment}_{int(plate.removeprefix('Plate'))}_{address}"


def base_key(key: str) -> str:
    return key.rsplit("_", 1)[0]


def read_metadata(path: Path) -> dict[str, tuple[str, int]]:
    metadata: dict[str, tuple[str, int]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            # The query-guide rows have a biological gene label and are the
            # cleanest same-gene retrieval subset in the public release.
            gene = (row.get("gene") or "").strip()
            if not gene or row.get("perturbation_type") != "CRISPR":
                continue
            if row.get("well_type_label") != "Query guides":
                continue
            metadata[row["well_id"]] = (gene, int(row["plate"]))
    if not metadata:
        raise RuntimeError(f"No CRISPR query-guide rows found in {path}")
    return metadata


def build_or_load_index(data_root: Path, cache_path: Path) -> list[WellRecord]:
    if cache_path.exists():
        with cache_path.open() as handle:
            return [WellRecord(**row) for row in map(json.loads, handle)]

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("RxRx3 indexing requires pyarrow") from exc

    metadata = read_metadata(data_root / "metadata_rxrx3_core.csv")
    shards = sorted((data_root / "data").glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No train parquet shards under {data_root / 'data'}")

    records: list[WellRecord] = []
    pending: list[tuple[str, int, int]] = []
    for shard_idx, shard_path in enumerate(shards):
        parquet = pq.ParquetFile(shard_path)
        row_offset = 0
        for row_group in range(parquet.num_row_groups):
            keys = parquet.read_row_group(row_group, columns=["__key__"]).column(0).to_pylist()
            for local_row, key in enumerate(keys):
                pending.append((key, shard_idx, row_offset + local_row))
                if len(pending) < 6:
                    continue
                first_key, first_shard, first_row = pending[0]
                expected = [f"{base_key(first_key)}_{channel}" for channel in range(1, 7)]
                observed = [item[0] for item in pending]
                if observed != expected:
                    raise RuntimeError(
                        "RxRx3 channel rows are not six consecutive ordered channels; "
                        f"first observed group={observed}"
                    )
                well_id = source_key_to_well_id(first_key)
                # A few parquet boundaries fall inside a six-channel well.
                # Skipping those rare records is safer than turning a shard
                # boundary into a silently wrong channel stack.
                same_shard = all(item[1] == first_shard for item in pending)
                if same_shard and well_id in metadata:
                    gene, plate = metadata[well_id]
                    records.append(
                        WellRecord(
                            key=base_key(first_key), well_id=well_id, gene=gene,
                            plate=plate, shard=first_shard, row=first_row,
                        )
                    )
                pending.clear()
            row_offset += len(keys)
        print(f"[index] {shard_idx + 1}/{len(shards)} shards; retained {len(records)} query-guide wells", flush=True)

    if pending:
        raise RuntimeError("A parquet shard ended in a partial six-channel group")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records


def choose_protocol_records(
    records: Iterable[WellRecord],
    max_eval_wells: int,
    max_calibration_wells: int,
    per_gene_cap: int,
) -> tuple[list[WellRecord], list[WellRecord], list[WellRecord]]:
    calibration, candidates = [], []
    for record in records:
        # Calibration and evaluation have disjoint gene identities.  The
        # projection never sees downstream labels while fitting, but this also
        # prevents transductive adaptation from inflating retrieval scores.
        (calibration if stable_bucket(record.gene, 5) == 0 else candidates).append(record)

    by_gene: dict[str, list[WellRecord]] = defaultdict(list)
    for record in candidates:
        by_gene[record.gene].append(record)

    gallery, query = [], []
    for gene in sorted(by_gene):
        rows = sorted(by_gene[gene], key=lambda row: (row.plate, row.well_id))
        unique_plates = sorted({row.plate for row in rows})
        if len(unique_plates) < 2:
            continue
        split = max(1, len(unique_plates) // 2)
        gallery_plates = set(unique_plates[:split])
        gene_gallery = [row for row in rows if row.plate in gallery_plates][:per_gene_cap]
        gene_query = [row for row in rows if row.plate not in gallery_plates][:per_gene_cap]
        if gene_gallery and gene_query:
            gallery.extend(gene_gallery)
            query.extend(gene_query)

    # Cap by complete label groups to preserve every query label in gallery.
    selected_genes = sorted({row.gene for row in gallery})
    max_genes = max(1, max_eval_wells // max(2, 2 * per_gene_cap))
    selected = set(selected_genes[:max_genes])
    gallery = [row for row in gallery if row.gene in selected]
    query = [row for row in query if row.gene in selected]
    calibration = sorted(calibration, key=lambda row: (row.shard, row.row))[:max_calibration_wells]
    if not gallery or not query or not calibration:
        raise RuntimeError("Protocol selection produced an empty calibration, gallery, or query set")
    return calibration, gallery, query


class ParquetWellReader:
    def __init__(self, data_root: Path, cache_groups: int = 12):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("RxRx3 reading requires pyarrow") from exc
        self.pq = pq
        self.shards = sorted((data_root / "data").glob("train-*.parquet"))
        self.files: dict[int, object] = {}
        self.group_cache: OrderedDict[tuple[int, int], dict[str, list]] = OrderedDict()
        self.cache_groups = cache_groups

    def _file(self, shard: int):
        if shard not in self.files:
            self.files[shard] = self.pq.ParquetFile(self.shards[shard])
        return self.files[shard]

    def _group(self, shard: int, group: int) -> dict[str, list]:
        key = (shard, group)
        if key in self.group_cache:
            self.group_cache.move_to_end(key)
            return self.group_cache[key]
        table = self._file(shard).read_row_group(group, columns=["__key__", "jp2"]).to_pydict()
        self.group_cache[key] = table
        while len(self.group_cache) > self.cache_groups:
            self.group_cache.popitem(last=False)
        return table

    def read(self, record: WellRecord) -> np.ndarray:
        # Public parquet row groups have 100 rows.  Derive this from metadata
        # rather than assuming it stays fixed across future dataset revisions.
        parquet = self._file(record.shard)
        offset, row_group = record.row, 0
        while offset >= parquet.metadata.row_group(row_group).num_rows:
            offset -= parquet.metadata.row_group(row_group).num_rows
            row_group += 1
        arrays = []
        current_group, local = row_group, offset
        for channel in range(1, 7):
            table = self._group(record.shard, current_group)
            key = table["__key__"][local]
            expected = f"{record.key}_{channel}"
            if key != expected:
                raise RuntimeError(f"Unexpected channel key {key!r}; expected {expected!r}")
            payload = table["jp2"][local]["bytes"]
            with Image.open(io.BytesIO(payload)) as image:
                arrays.append(np.asarray(image).astype(np.float32, copy=False))
            local += 1
            if local == len(table["__key__"]) and channel < 6:
                current_group += 1
                local = 0
        return np.stack(arrays, axis=0)


def batches(reader: ParquetWellReader, records: list[WellRecord], batch_size: int):
    records = sorted(records, key=lambda row: (row.shard, row.row))
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        arrays = [reader.read(record) for record in rows]
        if len({array.shape for array in arrays}) != 1:
            raise RuntimeError("RxRx3 selected wells have inconsistent image shapes")
        yield torch.from_numpy(np.stack(arrays)), rows


def robust_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-image/channel percentile normalization before any six-to-three map."""
    x = x.float()
    flat = x.flatten(2)
    lo = torch.quantile(flat, 0.01, dim=2, keepdim=True).unsqueeze(-1)
    hi = torch.quantile(flat, 0.99, dim=2, keepdim=True).unsqueeze(-1)
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0.0, 1.0)


class ConvProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(6, 3, kernel_size=1, bias=False)
        self.decoder = torch.nn.Conv2d(3, 6, kernel_size=1, bias=False)
        with torch.no_grad():
            self.encoder.weight.zero_()
            self.decoder.weight.zero_()
            for output in range(3):
                self.encoder.weight[output, 2 * output : 2 * output + 2, 0, 0] = 0.5
                self.decoder.weight[2 * output : 2 * output + 2, output, 0, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def fit_projection(
    reader: ParquetWellReader,
    calibration: list[WellRecord],
    device: torch.device,
    batch_size: int,
    steps: int,
    lr: float,
    cache_wells: int,
) -> ConvProjection:
    projection = ConvProjection().to(device)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=lr, weight_decay=1e-5)
    cache_rows = calibration[: min(len(calibration), cache_wells)]
    # Adapter fitting does not require the backbone.  Keeping this bounded
    # calibration set in CPU RAM avoids repeatedly decoding the same JP2
    # channels for every 1x1-convolution update.
    raw_cache = torch.cat([raw for raw, _ in batches(reader, cache_rows, batch_size)], dim=0)
    print(f"[adapter] cached {len(raw_cache)} calibration wells in CPU RAM", flush=True)
    projection.train()
    for step in range(1, steps + 1):
        draw = torch.randint(len(raw_cache), (batch_size,))
        x = robust_normalize(raw_cache[draw].to(device, non_blocking=True))
        reconstruction = projection.decoder(projection(x))
        loss = F.mse_loss(reconstruction, x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"[adapter] step={step}/{steps} mse={loss.item():.6f}", flush=True)
    return projection.eval()


def make_rgb(x: torch.Tensor, policy: str, projection: ConvProjection | None) -> torch.Tensor:
    x = robust_normalize(x)
    if policy == "first3":
        return x[:, :3]
    if policy == "compact3":
        return x.reshape(x.shape[0], 3, 2, *x.shape[-2:]).mean(dim=2)
    if policy == "conv6to3":
        if projection is None:
            raise ValueError("conv6to3 requires a fitted ConvProjection")
        return projection(x).clamp_(0.0, 1.0)
    raise ValueError(f"Unknown input policy={policy}")


def image_net_normalize(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_t = torch.tensor(mean[:3], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std[:3], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t.clamp_min(1e-6)


def layer_presets(backbone: torch.nn.Module) -> tuple[list[int], list[int]]:
    """Return the repository's official even-four and last-four tap sets."""
    depth = int(getattr(backbone, "n_blocks", len(getattr(backbone, "blocks"))))
    if depth == 12:
        even4 = [2, 5, 8, 11]
    elif depth == 24:
        even4 = [4, 11, 17, 23]
    elif depth == 32:
        even4 = [7, 15, 23, 31]
    elif depth == 40:
        even4 = [9, 19, 29, 39]
    else:
        even4 = [max(0, round((index + 1) * depth / 4) - 1) for index in range(4)]
    return even4, list(range(depth - 4, depth))


@torch.inference_mode()
def extract_readouts(
    reader: ParquetWellReader,
    records: list[WellRecord],
    backbone: torch.nn.Module,
    projection: ConvProjection | None,
    policy: str,
    device: torch.device,
    batch_size: int,
    image_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    all_features: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: list[str] = []
    even4, last4 = layer_presets(backbone)
    requested_layers = sorted(set(even4 + last4))
    for batch_index, (raw, batch_records) in enumerate(batches(reader, records, batch_size), 1):
        x = make_rgb(raw.to(device, non_blocking=True), policy, projection)
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
        x = image_net_normalize(x, mean, std)
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = backbone.get_intermediate_layers(
                x, n=requested_layers, reshape=False, return_class_token=True
            )
        token_by_layer = {layer: entry for layer, entry in zip(requested_layers, tokens)}
        even4_patch = [token_by_layer[layer][0].float() for layer in even4]
        even4_cls = [token_by_layer[layer][1].float() for layer in even4]
        last4_patch = [token_by_layer[layer][0].float() for layer in last4]
        last4_cls = [token_by_layer[layer][1].float() for layer in last4]
        final_patch, final_cls = token_by_layer[last4[-1]]
        final_patch, final_cls = final_patch.float(), final_cls.float()
        readouts = {
            "last1_cls": final_cls,
            "last1_cls_patch": torch.cat((final_cls, final_patch.mean(dim=1)), dim=1),
            # Means hold feature dimension fixed; concatenations quantify the
            # additional capacity available to a downstream linear probe.
            "even4_cls_mean": torch.stack(even4_cls, dim=0).mean(dim=0),
            "even4_cls_concat": torch.cat(even4_cls, dim=1),
            "last4_cls_mean": torch.stack(last4_cls, dim=0).mean(dim=0),
            "last4_cls_concat": torch.cat(last4_cls, dim=1),
        }
        for layer, cls in zip(even4, even4_cls):
            readouts[f"layer{layer}_cls"] = cls
        for name, feature in readouts.items():
            all_features[name].append(F.normalize(feature, dim=1).cpu().numpy().astype(np.float16))
        labels.extend(record.gene for record in batch_records)
        if batch_index == 1 or batch_index % 25 == 0:
            print(f"[features] {policy}: {min(batch_index * batch_size, len(records))}/{len(records)}", flush=True)
    return {key: np.concatenate(value).astype(np.float32) for key, value in all_features.items()}, np.asarray(labels)


def retrieval_metrics(gallery: np.ndarray, gallery_y: np.ndarray, query: np.ndarray, query_y: np.ndarray, device: torch.device) -> dict[str, float]:
    gallery_t = F.normalize(torch.from_numpy(gallery).to(device), dim=1)
    query_t = F.normalize(torch.from_numpy(query).to(device), dim=1)
    classes = {label: index for index, label in enumerate(sorted(set(gallery_y))) }
    gallery_labels = torch.tensor([classes[label] for label in gallery_y], device=device)
    query_labels = torch.tensor([classes[label] for label in query_y], device=device)
    topk = min(10, len(gallery_y))
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    for start in range(0, len(query_y), 256):
        similarity = query_t[start : start + 256] @ gallery_t.T
        ranks = similarity.topk(topk, dim=1).indices
        relevant = gallery_labels[ranks].eq(query_labels[start : start + len(ranks), None])
        for row in relevant.cpu().numpy():
            positives = np.flatnonzero(row)
            if len(positives):
                reciprocal += 1.0 / float(positives[0] + 1)
            for k in hits:
                hits[k] += int(np.any(row[: min(k, len(row))]))
    n_query = len(query_y)
    return {**{f"recall_at_{k}": hits[k] / n_query for k in hits}, "mrr_at_10": reciprocal / n_query}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--index-path", type=Path, default=None,
        help="Optional shared JSONL index, so repeated input/layer sweeps do not rescan all parquet shards.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-eval-wells", type=int, default=4096)
    parser.add_argument("--max-calibration-wells", type=int, default=2048)
    parser.add_argument("--per-gene-cap", type=int, default=3)
    parser.add_argument("--adapter-steps", type=int, default=400)
    parser.add_argument("--adapter-lrs", default="0.003,0.001")
    parser.add_argument("--adapter-cache-wells", type=int, default=256)
    parser.add_argument("--skip-adapter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This screen requires a CUDA device")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index_path or (args.output_dir / "rxrx3_crispr_query_guide_index.jsonl")
    records = build_or_load_index(args.data_root, index_path)
    calibration, gallery_records, query_records = choose_protocol_records(
        records, args.max_eval_wells, args.max_calibration_wells, args.per_gene_cap
    )
    print(
        f"[protocol] calibration={len(calibration)} gallery={len(gallery_records)} "
        f"query={len(query_records)} genes={len(set(row.gene for row in query_records))}", flush=True
    )
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "train_config": str(args.train_config),
                "data_root": str(args.data_root),
                "index_path": str(index_path),
                "calibration_wells": len(calibration),
                "gallery_wells": len(gallery_records),
                "query_wells": len(query_records),
                "genes": len({row.gene for row in query_records}),
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    reader = ParquetWellReader(args.data_root)
    mean, std = _load_multichannel_stats(args.train_config)
    backbone = load_dinov3_backbone(str(args.checkpoint), str(args.train_config), device=device, freeze=True).eval()

    projections: list[tuple[str, ConvProjection | None]] = [("first3", None), ("compact3", None)]
    if not args.skip_adapter:
        for lr_text in args.adapter_lrs.split(","):
            lr = float(lr_text)
            projection = fit_projection(
                reader, calibration, device, args.batch_size, args.adapter_steps, lr, args.adapter_cache_wells
            )
            tag = f"conv6to3_lr{lr:g}".replace(".", "p")
            torch.save({"state_dict": projection.state_dict(), "lr": lr, "steps": args.adapter_steps}, args.output_dir / f"{tag}.pth")
            projections.append((tag, projection))

    rows = []
    for policy_tag, projection in projections:
        policy = "conv6to3" if policy_tag.startswith("conv6to3") else policy_tag
        gallery_x, gallery_y = extract_readouts(
            reader, gallery_records, backbone, projection, policy, device, args.batch_size, args.image_size, mean, std
        )
        query_x, query_y = extract_readouts(
            reader, query_records, backbone, projection, policy, device, args.batch_size, args.image_size, mean, std
        )
        for readout in gallery_x:
            metrics = retrieval_metrics(gallery_x[readout], gallery_y, query_x[readout], query_y, device)
            rows.append({
                "input_policy": policy_tag, "readout": readout,
                "n_gallery": len(gallery_y), "n_query": len(query_y),
                "n_genes": len(set(query_y)), **metrics,
            })
            np.savez_compressed(args.output_dir / f"features_{policy_tag}_{readout}.npz", gallery=gallery_x[readout], gallery_y=gallery_y, query=query_x[readout], query_y=query_y)

    summary = args.output_dir / "summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
