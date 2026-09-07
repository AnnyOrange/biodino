from pathlib import Path

from scripts.build_expert_metadata_overlay import (
    catalog_metadata,
    record_files,
    recover_source_paths,
    source_id_from_key,
)


def test_source_id_from_key() -> None:
    assert source_id_from_key("shard.tar::id000001_oid420_crop_0_0") == 420


def test_record_recovery_and_catalog_metadata(tmp_path: Path) -> None:
    records = tmp_path / "records"
    records.mkdir()
    path = records / "tile.records.jsonl"
    path.write_text(
        '{"source_id": 7, "source_path": "/root/idr0001-demo/image.tif"}\n'
        '{"source_id": 8, "source_path": "/root/idr0002-demo/image.tif"}\n',
        encoding="utf-8",
    )

    files = record_files([records])
    recovered, stats = recover_source_paths({7, 9}, files, progress_gib=1.0)
    assert recovered == {7: "/root/idr0001-demo/image.tif"}
    assert stats["scanned_lines"] == 2

    catalog = {
        "idr:idr0001-demo": {
            "domain": "idr:idr0001-demo",
            "organism": "Homo sapiens",
            "acquisition_family": "fluorescence_microscopy",
            "sample_type": "cell",
        }
    }
    assert catalog_metadata("idr:idr0001-demo", catalog) == {
        "domain": "idr:idr0001-demo",
        "organism": "Homo sapiens",
        "acquisition_family": "fluorescence_microscopy",
        "sample_type": "cell",
    }
