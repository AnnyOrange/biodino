import psycopg2
import json
import os
import time
import math
from collections import defaultdict

# --- 1. 配置 (Configuration) ---
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "pW5^sL9&hJmZ1!uN",
    "host": "172.16.0.217",
    "port": "5432"
}

# 路径配置
OUTPUT_DIR = "/mnt/huawei_deepcad/onepb/shards_task_highNA"
REPORT_FILE = "task2_hybrid_report_highNA.txt"

# --- 额外过滤：排除明显非真实数据/实验/代码目录 ---
# 说明：使用 Postgres 正则（!~*）做 case-insensitive 排除
EXCLUDE_PATTERN = (
    r"(SRDTrans-main|Simulation|Sparse2pSAM_Simulation|reconstruction|DeepCAD|for_denoise|Unet|实验|"
    r"denoise|simu|重建代码|unet|rcan|0-JYS/lf_restore/experiments|0-WXC/ImagePro|0-WXC/CY|rec|"
    r"denoising|serenet|epoch|fake|Reconstruction|codeRelease|labviewtest|biastest|Field-angle-energyTest|"
    r"beads|0-WSD/20250731_test|0-WSD/20250417_PMTdata|Calibrate|20250222_offsetData|systemCalibrate|"
    r"Large2pSAM_AngleTest|flutest|Large2pSAM_ResolutionTest|0-WSD/20250416_PMTdata|biderectionTest|"
    r"2pCOSMIC_systemTest|offsetTest|2pCOSM_SystemTest|system-test|maichong|zScan|yingguang|0-WSD/20250701|"
    r"LEAO/Fig_HighOrderZern|0-WXC/center_view0316|bead|PSF|psf|Zscan|calibration)"
)

# --- 2. 核心参数 (Core Parameters) ---
# 【重要】现在的 SHARD_SIZE 指的是"帧数/Frame Count"，即 JSON list 的长度
# 因为我们把 stack 展开了，这个数字直接对应 DataLoader 里的 batch item 数量
MAX_SHARD_SIZE = 20000  

MAX_PIXEL_EDGE = 5000 
MIN_PIXEL_EDGE = 64     

# 分组阈值 (保持不变，依然基于总帧数判断是否为 Standard Group)
MIN_GROUP_SIZE = 100    

# 分桶阈值
BUCKET_SIZE_THRESH = {
    'small': 256,
    'medium': 1024,
    'large': 2048,
    'huge': MAX_PIXEL_EDGE
}
BUCKET_AR_THRESH = {
    'portrait': 0.8,
    'square': 1.2
}

DRY_RUN = False

if not DRY_RUN:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. 辅助函数 (Helper Functions) ---

def get_sql_query():
    """
    SQL 查询 - 针对 highNA 数据
    筛选条件：
    1. file_path 包含 'highNA'
    2. 按 channel_count 分桶（不再强制 channel_count=1）
    3. h, w 在合理范围内
    """
    # 注意：在 execute 参数化查询中，字面量 % 必须写成 %%
    base_query = f"""
    SELECT 
        0 as dataset_id,
        id as original_id, 
        file_path,
        h,
        w,
        count,
        channel_count
    FROM original_image_all_2p_parsed
    WHERE file_path LIKE '%%highNA%%'
      AND file_path !~* %s
      AND file_path NOT LIKE '%%.zip'
      AND h >= {MIN_PIXEL_EDGE}
      AND w >= {MIN_PIXEL_EDGE}
      AND h <= {MAX_PIXEL_EDGE}
      AND w <= {MAX_PIXEL_EDGE}
    ORDER BY h, w, channel_count
    """
    return base_query

def get_hybrid_bucket_key(h, w):
    max_edge = max(h, w)
    size_key = 'huge'
    if max_edge <= BUCKET_SIZE_THRESH['small']: size_key = 'small'
    elif max_edge <= BUCKET_SIZE_THRESH['medium']: size_key = 'medium'
    elif max_edge <= BUCKET_SIZE_THRESH['large']: size_key = 'large'
        
    if h == 0: ar = 999
    else: ar = w / h
    
    shape_key = 'landscape'
    if ar <= BUCKET_AR_THRESH['portrait']: shape_key = 'portrait'
    elif ar < BUCKET_AR_THRESH['square']: shape_key = 'square'
        
    return f"{size_key}_{shape_key}"

def write_shard_file(data, meta, filename):
    if DRY_RUN: return
    filepath = os.path.join(OUTPUT_DIR, filename)
    payload = { "meta": meta, "files": data }
    with open(filepath, 'w') as f:
        json.dump(payload, f)

# --- 4. 主函数 (Main Execution) ---

def main():
    start_time = time.time()
    
    # --- 准备 buffer ---
    # Value 存储结构: [(path, count, h, w, channel_idx, channel_count), ...]
    # 增加 channel 维度到 key 中: (h, w, channel_count, channel_idx)
    standard_groups = {} 
    misc_buckets = defaultdict(list) 
    
    stats = {
        "total_rows_read": 0,
        "total_frames_generated": 0,  # 展开后的总帧数
        "total_filtered_out_unsafe": 0,
        "standard_groups_count": 0,
        "standard_frames_count": 0,
        "misc_frames_count_by_bucket": defaultdict(int),
        "total_shard_files": 0
    }
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f_log:
        def log(message, force_print=True):
            if force_print: print(message)
            f_log.write(message + "\n")

        log(f"Mode: {'DRY RUN' if DRY_RUN else 'LIVE (WRITING FILES)'}")
        log(f"Strategy: OPTION C - VIRTUAL SPLITTING (Frame Indexing)")
        log(f"Output Dir: {OUTPUT_DIR if not DRY_RUN else 'N/A'}")
        log(f"Report File: {os.path.abspath(REPORT_FILE)}")
        log("--- DATA SOURCE ---")
        log(f"  Table: original_image_all_2p_parsed")
        log(f"  Filters:")
        log(f"    - file_path LIKE '%highNA%'")
        log(f"    - file_path !~* EXCLUDE_PATTERN (long regex)")
        log(f"    - file_path NOT LIKE '%.zip'")
        log(f"    - group by (h, w, channel_count, channel_idx) [channel-splitting]")
        log("--- THRESHOLDS ---")
        log(f"  Safety Filter: < {MIN_PIXEL_EDGE}px or > {MAX_PIXEL_EDGE}px (DROPPED)")
        log(f"  Standard Group (Head): >= {MIN_GROUP_SIZE} frames")
        log(f"  Max Shard Size: {MAX_SHARD_SIZE} frames")
        log("-" * 60)

        try:
            log(f"Connecting to DB at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(name='task2_highNA_split_cursor')
            # 传递参数元组，注意 EXCLUDE_PATTERN 需要放在 tuple 中
            cursor.execute(get_sql_query(), (EXCLUDE_PATTERN,))
        except Exception as e:
            log(f"❌ DB Error: {e}")
            return

        # ============================================
        # PHASE 1: READ & CLASSIFY (In-Memory Buffer)
        # ============================================
        log("Phase 1: Reading from DB and classifying...")
        
        current_group_key = None
        # Buffer item: (f_path, img_count, h, w, channel_idx, channel_count)
        current_group_buffer = []
        current_group_frame_count = 0 

        while True:
            rows = cursor.fetchmany(5000)
            if not rows: break
                
            for row in rows:
                stats["total_rows_read"] += 1
                if stats["total_rows_read"] % 100000 == 0:
                    log(f" ... read {stats['total_rows_read'] // 1000}K rows...", force_print=True)
                
                # 必须解包 7 个字段
                d_id, _, f_path, h, w, img_count, channel_count = row
                
                img_count = img_count or 1
                channel_count = channel_count or 1
                if channel_count < 1: channel_count = 1
                
                # 安全过滤
                if h > MAX_PIXEL_EDGE or w > MAX_PIXEL_EDGE or h < MIN_PIXEL_EDGE or w < MIN_PIXEL_EDGE:
                    stats["total_filtered_out_unsafe"] += img_count * channel_count
                    continue

                # === 关键：按 channel 分桶 ===
                # 将同一个文件按 channel_idx 拆成多个逻辑任务，避免 C1/C2 混在一起
                for channel_idx in range(int(channel_count)):
                    # Group Key 包含通道索引，确保不同通道的数据完全分开
                    group_key = (h, w, int(channel_count), int(channel_idx))
                
                    # --- Group Logic ---
                    if group_key != current_group_key:
                        if current_group_buffer:
                            # Process previous group
                            prev_h, prev_w, prev_c, prev_ch_idx = current_group_key
                        
                            if current_group_frame_count >= MIN_GROUP_SIZE:
                                standard_groups[(prev_h, prev_w, prev_c, prev_ch_idx)] = current_group_buffer
                                stats["standard_groups_count"] += 1
                                stats["standard_frames_count"] += current_group_frame_count
                            else:
                                bucket_key = get_hybrid_bucket_key(prev_h, prev_w)
                                # Misc bucket key 也要区分通道
                                bucket_key = f"{bucket_key}_c{prev_c}_ch{prev_ch_idx+1}"
                                misc_buckets[bucket_key].extend(current_group_buffer)
                                stats["misc_frames_count_by_bucket"][bucket_key] += current_group_frame_count
                                
                        # Reset
                        current_group_buffer = [(f_path, img_count, h, w, channel_idx, int(channel_count))]
                        current_group_frame_count = img_count
                        current_group_key = group_key
                    else:
                        current_group_buffer.append((f_path, img_count, h, w, channel_idx, int(channel_count)))
                        current_group_frame_count += img_count
        
        # Process last group
        if current_group_buffer:
            prev_h, prev_w, prev_c, prev_ch_idx = current_group_key
            if current_group_frame_count >= MIN_GROUP_SIZE:
                standard_groups[(prev_h, prev_w, prev_c, prev_ch_idx)] = current_group_buffer
                stats["standard_groups_count"] += 1
                stats["standard_frames_count"] += current_group_frame_count
            else:
                bucket_key = get_hybrid_bucket_key(prev_h, prev_w)
                bucket_key = f"{bucket_key}_c{prev_c}_ch{prev_ch_idx+1}"
                misc_buckets[bucket_key].extend(current_group_buffer)
                stats["misc_frames_count_by_bucket"][bucket_key] += current_group_frame_count

        cursor.close()
        conn.close()
        log(f"Phase 1 Complete. Total frames tracked: {stats['standard_frames_count'] + sum(stats['misc_frames_count_by_bucket'].values())}")

        # ============================================
        # PHASE 2: WRITE SHARDS (WITH VIRTUAL SPLITTING)
        # ============================================
        log(f"Phase 2: Writing shards with Virtual Splitting (expanding counts to frame indices)...")
        
        def write_expanded_shards(entries, shard_type, group_h, group_w, bucket_key=None):
            """
            entries: list of (file_path, count, h, w, channel_idx, channel_count)
            这里我们将每个 file 展开成 count 个 item
            """
            nonlocal stats
            
            current_shard_items = []
            
            # 遍历每一个文件
            for f_path, f_count, f_h, f_w, channel_idx, channel_count in entries:
                
                # 【关键】展开 Loop
                # 将 1 个文件记录展开为 f_count 个帧记录
                for i in range(f_count):
                    
                    # 构建 Item
                    item = {
                        "path": f_path,
                        "frame_idx": i, # 核心：帧索引
                        "original_size": [f_h, f_w],
                        # === highNA 关键：通道信息 ===
                        "channel_idx": int(channel_idx),  # 0-based
                        "expected_channel_count": int(channel_count),
                    }
                    
                    current_shard_items.append(item)
                    
                    # 检查是否满了
                    if len(current_shard_items) >= MAX_SHARD_SIZE:
                        # 写入 Shard
                        if shard_type == "standard":
                            # 文件名包含通道信息
                            fname = f"task_{stats['total_shard_files']:06d}_std_h{group_h}_w{group_w}_ch{channel_idx+1}.json"
                            meta_h, meta_w = group_h, group_w
                        else:
                            fname = f"task_{stats['total_shard_files']:06d}_misc_{bucket_key}.json"
                            meta_h, meta_w = -1, -1 
                        
                        meta = {
                            "type": shard_type,
                            "bucket": f"{group_h}x{group_w}" if shard_type == "standard" else bucket_key,
                            "height": meta_h,
                            "width": meta_w,
                            "count": len(current_shard_items),
                            "is_2p": True,  # 标记这是 2P 数据
                            "channel_idx": int(channel_idx),
                            "expected_channel_count": int(channel_count),
                        }
                        
                        write_shard_file(current_shard_items, meta, fname)
                        stats["total_shard_files"] += 1
                        stats["total_frames_generated"] += len(current_shard_items)
                        
                        # 清空 buffer
                        current_shard_items = []
            
            # 写入剩余的
            if current_shard_items:
                # 获取第一条数据的 channel 信息用于命名（同一个 shard 里 channel 应该是一致的，因为我们按 channel 分组了）
                first_ch_idx = current_shard_items[0]["channel_idx"]
                first_ch_cnt = current_shard_items[0]["expected_channel_count"]

                if shard_type == "standard":
                    fname = f"task_{stats['total_shard_files']:06d}_std_h{group_h}_w{group_w}_ch{first_ch_idx+1}.json"
                    meta_h, meta_w = group_h, group_w
                else:
                    fname = f"task_{stats['total_shard_files']:06d}_misc_{bucket_key}.json"
                    meta_h, meta_w = -1, -1
                
                meta = {
                    "type": shard_type,
                    "bucket": f"{group_h}x{group_w}" if shard_type == "standard" else bucket_key,
                    "height": meta_h,
                    "width": meta_w,
                    "count": len(current_shard_items),
                    "is_2p": True,
                    "channel_idx": int(first_ch_idx),
                    "expected_channel_count": int(first_ch_cnt),
                }
                write_shard_file(current_shard_items, meta, fname)
                stats["total_shard_files"] += 1
                stats["total_frames_generated"] += len(current_shard_items)

        # 2a. 写入 Standard Groups
        log(f" ... processing {stats['standard_groups_count']} Standard groups...")
        for (h, w, c, ch_idx), entries in standard_groups.items():
            write_expanded_shards(entries, "standard", h, w)

        # 2b. 写入 Misc Buckets
        log(f" ... processing {len(misc_buckets)} Misc buckets...")
        for bucket_key, entries in misc_buckets.items():
            write_expanded_shards(entries, "misc", -1, -1, bucket_key)
            
        log("Phase 2 Complete.")

        # ============================================
        # PHASE 3: REPORT
        # ============================================
        end_time = time.time()
        log("\n" + "="*60)
        log(f"HIGHNA SPLIT SHARDING SUMMARY")
        log("="*60)
        log(f"Total Time: {end_time - start_time:.2f} seconds")
        log(f"Total Rows (Files) Read: {stats['total_rows_read']}")
        log(f"Total Frames (Virtual Items) Generated: {stats['total_frames_generated']}")
        log(f"Total Shard Files: {stats['total_shard_files']}")
        log("-" * 60)
        log(f"Standard Groups: {stats['standard_groups_count']}")
        log(f"Standard Frames: {stats['standard_frames_count']}")
        log(f"Misc Frames: {sum(stats['misc_frames_count_by_bucket'].values())}")
        log("="*60)
        
    print(f"\nDone! Report generated: {REPORT_FILE}")

if __name__ == "__main__":
    main()