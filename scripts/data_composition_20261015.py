"""Sample per-tile metadata from each training pool (1TB, 5TB, route2 r0/r9) and tabulate what the extra 15TB actually adds:
source dataset, imaging method, channel structure, and how many tiles come from one source image / one specimen."""
import glob, json, random, sys, tarfile, collections, csv, re
from pathlib import Path
POOLS = {
    "1tb": "/mnt/huawei_deepcad/webds_micro_100k_by_channel_patched_shuffle/filtered_mixed_train_w*.tar",
    "5tb": "/mnt/huawei_blm/deepcad_5t_v1/wds_patched_shuffle/filtered_mixed_train*.tar",
    "route2_r0": "/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923/route2_15tb_no_old5_micro_tars/filtered_projection_20TB_nested-r0*.tar",
    "route2_r9": "/mnt/deepcad_nfs/deepcad_100t/final-data/20TB_takeover_20260922_2350/route2_strict_pathology_20260923/route2_15tb_no_old5_micro_tars/filtered_projection_20TB_nested-r9*.tar",
}
OUT = Path(sys.argv[1]); N_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40; PER = int(sys.argv[3]) if len(sys.argv) > 3 else 300
random.seed(0)
rows = []
for pool, pat in POOLS.items():
    shards = sorted(glob.glob(pat)); pick = random.sample(shards, min(N_SHARDS, len(shards)))
    for sh in pick:
        n = 0; nimg = collections.Counter()
        try:
            with tarfile.open(sh, "r|") as tf:
                for m in tf:
                    if m.name.endswith(".tif"): nimg[m.name.rsplit(".", 2)[0]] += 1
                    if not m.name.endswith(".meta.json"): continue
                    d = json.load(tf.extractfile(m)); key = m.name[: -len(".meta.json")]
                    path = d.get("source_path") or d.get("original_path") or ""
                    ds = d.get("source_dataset") or d.get("dataset_name") or ""
                    if not ds:
                        mm = re.search(r"/(idr\d{4}[^/]*|\d+-[^/]+)/", path); ds = mm.group(1) if mm else "?"
                    rows.append(dict(pool=pool, shard=Path(sh).name, key=key, dataset=ds,
                                     method=d.get("method_label", ""), stratum=d.get("selection_stratum", ""),
                                     src_image=path, channel_count=d.get("channel_count", d.get("source_channel_count", "")),
                                     tif_members=nimg.get(key, 0), score=d.get("final_score", "")))
                    n += 1
                    if n >= PER: break
        except Exception as e:
            print("ERR", sh, e, file=sys.stderr)
    print(pool, "rows", sum(r["pool"] == pool for r in rows), flush=True)
with open(OUT / "sampled_tiles.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for pool in POOLS:
    rs = [r for r in rows if r["pool"] == pool]
    c = collections.Counter(r["dataset"] for r in rs); m = collections.Counter(r["method"] or r["stratum"] for r in rs)
    imgs = collections.Counter(r["src_image"] for r in rs)
    print(f"\n== {pool}: {len(rs)} tiles, {len(c)} datasets, {len(imgs)} distinct source images (tiles/img {len(rs)/max(1,len(imgs)):.2f})")
    for k, v in c.most_common(12): print(f"   {v/len(rs):6.1%}  {k}")
    print("   methods:", ", ".join(f"{k}:{v/len(rs):.0%}" for k, v in m.most_common(6)))
    print("   channels:", collections.Counter(str(r["channel_count"]) for r in rs).most_common(4))
