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

# 基础路径，具体文件夹会在代码中动态生成
# 最终结构示例: 
#   /mnt/huawei_deepcad/onepb/shards_task_huge_ch1/target_ch0/
#   /mnt/huawei_deepcad/onepb/shards_task_huge_ch3/target_ch1/
BASE_OUTPUT_PATH = "/mnt/huawei_deepcad/onepb/shards_task_huge"
REPORT_FILE = "task_huge_multich_report.txt"

# [修改] 需要处理的通道列表: 1 到 13 通道
TARGET_CHANNELS = list(range(1, 14))

# --- 2. 核心参数 (Core Parameters) ---
MAX_SHARD_SIZE = 5000  # 大图元数据处理稍慢，每个 shard 放少一点方便并行

# 2a. 尺寸阈值 (Size Thresholds) - 针对超大图
MIN_LONG_EDGE = 5000   # 长边必须 > 5000
MIN_SHORT_EDGE = 1024  # 短边必须 > 1024

# 2b. 分组逻辑
MIN_GROUP_SIZE = 50    # 大图完全相同的概率较低，降低标准组阈值

# 2c. 宽高比桶 (Aspect Ratio Buckets)
BUCKET_AR_THRESH = {
    'portrait': 0.8, # AR <= 0.8
    'square': 1.2    # 0.8 < AR < 1.2
}

DRY_RUN = False

# --- 3. 辅助函数 (Helper Functions) ---

def parse_image_shape(shape_str, channel_count):
    """
    智能解析 image_shape 字符串，提取真实的 H 和 W。
    兼容 (H, W), (C, H, W), (H, W, C) 以及带 Batch 或 Z 轴的情况。
    """
    if not shape_str:
        return 0, 0
        
    try:
        dims = [int(p.strip()) for p in shape_str.strip('()').split(',') if p.strip()]
        
        if len(dims) < 2:
            return 0, 0
            
        # 单通道情况: 严格为 (H, W) 或 (1, H, W)
        if channel_count == 1:
            if len(dims) == 2:
                return dims[0], dims[1]
            elif 1 in dims:
                dims.remove(1) 
                
        # 多通道情况: 剔除一个等于 channel_count 的维度
        elif channel_count > 1:
            if channel_count in dims:
                dims.remove(channel_count) 
                
        # 剩下的维度中，取前两个最大值作为 H 和 W
        if len(dims) >= 2:
            sorted_dims = sorted(dims, reverse=True)
            return sorted_dims[0], sorted_dims[1]
        else:
            return 0, 0
            
    except Exception:
        return 0, 0

def get_sql_query(target_channel):
    """构建带过滤条件的极简 SQL"""
    filters = """
      AND s.file_path NOT LIKE '%mask%'
      AND s.file_path NOT LIKE '%flow%'
      AND s.file_path NOT LIKE '%idr0065%'
      AND s.file_path NOT LIKE '%labels%'
      AND s.file_path NOT LIKE '%#GT%'
      AND s.file_path NOT LIKE '%idr0157%'
    """
    
    query = f"""
    SELECT 
        s.file_path,
        s.image_shape
    FROM original_images_all s
    WHERE s.channel_count = {target_channel}
    {filters}
    """
    return query

def get_hybrid_bucket_key(h, w):
    """根据 H, W 返回混合策略的桶 Key (大图统一为 huge)"""
    size_key = 'huge'
    if h == 0: ar = 999
    else: ar = w / h
    
    shape_key = 'landscape' 
    if ar <= BUCKET_AR_THRESH['portrait']:
        shape_key = 'portrait'
    elif ar < BUCKET_AR_THRESH['square']:
        shape_key = 'square'
        
    return f"{size_key}_{shape_key}"

def write_shard_file(data, meta, filename, target_dir):
    """写入 JSON"""
    if DRY_RUN:
        return
        
    filepath = os.path.join(target_dir, filename)
    payload = { "meta": meta, "files": data }
    
    with open(filepath, 'w') as f:
        json.dump(payload, f)

# --- 4. 单通道处理逻辑 ---

def process_single_channel(channel_num, log_func):
    """处理某个固定总通道数的大图，并裂变为多个提取任务"""
    # 专属大图目录: shards_task_huge_ch{N}
    channel_dir_name = f"shards_task_huge_ch{channel_num}"
    current_output_dir = os.path.join(BASE_OUTPUT_PATH, channel_dir_name)
    
    log_func(f"\n>>> Processing HUGE Images for {channel_num}-Channel")
    log_func(f"    Base Dir: {current_output_dir}")
    
    if not DRY_RUN:
        os.makedirs(current_output_dir, exist_ok=True)

    standard_groups = {} 
    misc_buckets = defaultdict(list)
    stats = {
        "rows_read": 0,
        "filtered_size": 0,
        "std_count": 0,    
        "misc_count": 0,   
        "shards_created": 0 
    }

    # ============================================
    # Phase 1: Reading, Filtering & Grouping
    # ============================================
    groups_h_w = defaultdict(list)
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(name=f'cursor_huge_ch{channel_num}')
        
        sql = get_sql_query(channel_num)
        cursor.execute(sql)
        
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
                
            for row in rows:
                stats["rows_read"] += 1
                if stats["rows_read"] % 500000 == 0:
                     print(f"    [Huge Ch{channel_num}] Scanned {stats['rows_read']} rows...")

                f_path, shape_str = row
                
                # 智能提取真实的 H 和 W
                h, w = parse_image_shape(shape_str, channel_num)
                
                # --- 大图专属尺寸过滤逻辑 ---
                long_edge = max(h, w)
                short_edge = min(h, w)
                
                # 如果长边不够长，或者短边太短，丢弃（不属于大图范畴）
                if long_edge <= MIN_LONG_EDGE or short_edge <= MIN_SHORT_EDGE:
                    stats["filtered_size"] += 1
                    continue
                
                groups_h_w[(h, w)].append(f_path)

        cursor.close()
        conn.close()
        
        # 分流
        for (h, w), paths in groups_h_w.items():
            if len(paths) >= MIN_GROUP_SIZE:
                standard_groups[(h, w)] = paths
            else:
                b_key = get_hybrid_bucket_key(h, w)
                misc_buckets[b_key].extend(paths)
                
    except Exception as e:
        log_func(f"❌ Error processing {channel_num}-channel huge images: {e}")
        return stats

    # ============================================
    # Phase 2: Writing & Task Explosion
    # ============================================
    log_func(f"    Exploding {channel_num}-channel huge data into {channel_num} independent target tasks...")

    for target_c_idx in range(channel_num):
        # 为当前提取目标创建子文件夹
        specific_output_dir = os.path.join(current_output_dir, f"target_ch{target_c_idx}")
        if not DRY_RUN:
            os.makedirs(specific_output_dir, exist_ok=True)
            
        # 1. 写入标准组
        for (h, w), paths in standard_groups.items():
            if target_c_idx == 0:
                stats["std_count"] += len(paths)
                
            for i in range(0, len(paths), MAX_SHARD_SIZE):
                chunk = paths[i:i+MAX_SHARD_SIZE]
                
                filename = f"task_huge_{channel_num}CH_target{target_c_idx}_{stats['shards_created']:06d}_std_h{h}_w{w}.json"
                
                meta = {
                    "type": "standard_huge",
                    "total_channels": channel_num,
                    "extract_channel_idx": target_c_idx, # [核心指令]
                    "bucket": f"{h}x{w}",
                    "height": h,
                    "width": w,
                    "count": len(chunk)
                }
                write_shard_file(chunk, meta, filename, specific_output_dir)
                stats["shards_created"] += 1

        # 2. 写入 Misc 桶
        for b_key, paths in misc_buckets.items():
            if target_c_idx == 0:
                stats["misc_count"] += len(paths)
                
            for i in range(0, len(paths), MAX_SHARD_SIZE):
                chunk = paths[i:i+MAX_SHARD_SIZE]
                
                filename = f"task_huge_{channel_num}CH_target{target_c_idx}_{stats['shards_created']:06d}_misc_{b_key}.json"
                
                meta = {
                    "type": "misc_huge",
                    "total_channels": channel_num,
                    "extract_channel_idx": target_c_idx, # [核心指令]
                    "bucket": b_key,
                    "height": -1,
                    "width": -1,
                    "count": len(chunk)
                }
                write_shard_file(chunk, meta, filename, specific_output_dir)
                stats["shards_created"] += 1
                
    log_func(f"    Finished. Base Huge Images: {stats['std_count'] + stats['misc_count']}, Filtered (Too small): {stats['filtered_size']}, Shards Created: {stats['shards_created']}")
    return stats

# --- 5. 主程序 ---

def main():
    start_total = time.time()
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f_log:
        def log(message, force_print=True):
            if force_print: print(message)
            f_log.write(message + "\n")
            
        log(f"Multi-Channel HUGE Image Task Explosion Sharding")
        log(f"Base Output Path: {BASE_OUTPUT_PATH}")
        log(f"Target Channels: {TARGET_CHANNELS}")
        log(f"Size Filter: Long Edge > {MIN_LONG_EDGE} AND Short Edge > {MIN_SHORT_EDGE}")
        log("="*60)
        
        grand_total_images = 0
        grand_total_shards = 0
        
        for ch in TARGET_CHANNELS:
            ch_stats = process_single_channel(ch, log)
            
            grand_total_images += (ch_stats['std_count'] + ch_stats['misc_count'])
            grand_total_shards += ch_stats['shards_created']
            f_log.flush()

        end_total = time.time()
        log("\n" + "="*60)
        log("GRAND TOTAL SUMMARY")
        log("="*60)
        log(f"Total Time: {end_total - start_total:.2f}s")
        log(f"Total Unique Physical HUGE Images: {grand_total_images}")
        log(f"Total Exploded Shards Created: {grand_total_shards}")
        log("="*60)

if __name__ == "__main__":
    main()