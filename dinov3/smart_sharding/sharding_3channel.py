import psycopg2
import json
import os
import time
import math
from collections import defaultdict

# --- 1. 配置 (Configuration) ---
# 支持从环境变量读取数据库配置，如果没有设置则使用默认值
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "pW5^sL9&hJmZ1!uN",
    "host": "172.16.0.217",
    "port": "5432"
}

# 路径配置
OUTPUT_DIR = "/mnt/huawei_deepcad/onepb/shards_task_3ch"
REPORT_FILE = "task2_hybrid_report_3ch.txt"  # 详细的统计报告

# --- 2. 核心参数 (Core Parameters) ---
MAX_SHARD_SIZE = 20000  # 每个 JSON 文件的最大图片数
EXCLUDE_PRIMARY = False

# 2a. 安全过滤 (直接丢弃)
MAX_PIXEL_EDGE = 5000  # 最长边 > 5k 
MIN_PIXEL_EDGE = 64     # 最短边 < 64

# 2b. 路径排除规则 - 低质量数据过滤
PATH_EXCLUSIONS = [
    '%mask%',
    '%flow%', 
    '%idr0065%',
    '%labels%',
    '%#GT%',
    '%idr0157%'
]

# 2d. "头部"与"尾部"的分割线
MIN_GROUP_SIZE = 100    # 分辨率组的图片数 >= 100，才算作"标准组"

# 2e. "尾部"分桶阈值
# 尺寸桶 (Size Buckets) - 基于最长边
BUCKET_SIZE_THRESH = {
    'small': 256,
    'medium': 1024,
    'large': 2048,
    'huge': MAX_PIXEL_EDGE # 2048 以上到安全阈值
}
# 宽高比桶 (Aspect Ratio Buckets) - (width / height)
BUCKET_AR_THRESH = {
    'portrait': 0.8, # AR <= 0.8
    'square': 1.2    # 0.8 < AR < 1.2
    # AR >= 1.2 自动归为 landscape
}

# 控制开关: True = 只生成报告，不写文件。False = 正式写入 JSON。
DRY_RUN = False

if not DRY_RUN:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. 辅助函数 (Helper Functions) ---

def parse_image_shape(shape_str):
    """
    解析 image_shape 字符串，提取 H 和 W。
    
    对于 3 通道图像，shape 可能是:
      - (H, W, 3) - channel-last 格式
      - (3, H, W) - channel-first 格式
    
    策略：找到所有维度，排除值为 3 的那个，剩下的两个就是 H 和 W。
    假设 H 和 W 都大于 3（实际图像尺寸通常远大于 3）。
    
    如果有多个维度等于 3，或者维度数不对，返回 None, None。
    """
    try:
        # 去掉括号，分割
        cleaned = shape_str.strip().strip('()')
        parts = [int(x.strip()) for x in cleaned.split(',')]
        
        if len(parts) == 2:
            # 只有两个维度，直接返回 (H, W)
            return parts[0], parts[1]
        
        if len(parts) == 3:
            # 三个维度，找出不是 3 的两个
            non_three = [p for p in parts if p != 3]
            
            if len(non_three) == 2:
                # 正常情况：一个维度是 3，另外两个是 H 和 W
                # 判断 channel 在哪里来确定 H/W 顺序
                if parts[2] == 3:
                    # (H, W, 3) - channel-last
                    return parts[0], parts[1]
                elif parts[0] == 3:
                    # (3, H, W) - channel-first
                    return parts[1], parts[2]
                else:
                    # (H, 3, W) - 中间是 3，不太常见，假设是 (H, W)
                    return parts[0], parts[2]
            
            elif len(non_three) == 3:
                # 没有维度等于 3（可能是 channel_count 不是 3？）
                # 假设最后一个是 channel，取前两个
                return parts[0], parts[1]
            
            elif len(non_three) == 1:
                # 有两个维度等于 3（如 3x3xC），这种图太小，应该被过滤
                return 3, non_three[0]
            
            else:
                # 三个维度都是 3，太小了
                return 3, 3
        
        if len(parts) == 4:
            # 可能是 (B, H, W, C) 或 (B, C, H, W)，取中间两个
            # 排除值为 3 的维度
            non_three = [p for p in parts if p != 3]
            if len(non_three) >= 2:
                return non_three[0], non_three[1]
            return None, None
        
        return None, None
    
    except Exception as e:
        return None, None


def get_sql_query():
    """
    构建核心 SQL，针对 3 通道图像，同时排除低质量数据。
    
    条件：
    1. channel_count = 3
    2. 排除路径包含: mask, flow, idr0065, labels, #GT, idr0157
    
    注意：直接返回 image_shape 原始字符串，由 Python 解析 H/W。
    """
    # 构建路径排除条件
    path_exclusion_clauses = " AND ".join([
        f"s.file_path NOT LIKE '{pattern}'" for pattern in PATH_EXCLUSIONS
    ])
    
    base_query = f"""
    SELECT 
        0 as dataset_id,
        s.id as original_id, 
        s.file_path,
        s.image_shape
    FROM original_images_all s
    WHERE s.channel_count = 3
      AND {path_exclusion_clauses}
    """

    if EXCLUDE_PRIMARY:
        final_query = f"""
        {base_query}
        AND NOT EXISTS (
            SELECT 1 
            FROM "primary" p 
            WHERE p.dataset = 0 
              AND p.original_id = s.original_id
        )
        """
    else:
        final_query = base_query
    
    return final_query


def get_hybrid_bucket_key(h, w):
    """
    根据 H, W 返回混合策略的桶 Key, 例如: "large_portrait"
    """
    # 1. 确定尺寸桶 (Size Bucket)
    max_edge = max(h, w)
    size_key = 'huge' # 默认
    if max_edge <= BUCKET_SIZE_THRESH['small']:
        size_key = 'small'
    elif max_edge <= BUCKET_SIZE_THRESH['medium']:
        size_key = 'medium'
    elif max_edge <= BUCKET_SIZE_THRESH['large']:
        size_key = 'large'
        
    # 2. 确定形状桶 (Shape Bucket)
    if h == 0: ar = 999 # 防止除零
    else: ar = w / h
    
    shape_key = 'landscape' # 默认
    if ar <= BUCKET_AR_THRESH['portrait']:
        shape_key = 'portrait'
    elif ar < BUCKET_AR_THRESH['square']:
        shape_key = 'square'
        
    return f"{size_key}_{shape_key}"


def write_shard_file(data, meta, filename):
    """通用的JSON写入函数"""
    if DRY_RUN:
        return # Dry run 模式不写文件
        
    filepath = os.path.join(OUTPUT_DIR, filename)
    payload = { "meta": meta, "files": data }
    
    with open(filepath, 'w') as f:
        json.dump(payload, f) # 非 pretty-print，节省磁盘IO和空间


# --- 4. 主函数 (Main Execution) ---

def main():
    start_time = time.time()
    
    # --- 准备所有桶 ---
    standard_groups = {}  # 字典: Key=(h,w), Value=[paths...]
    misc_buckets = defaultdict(list) # 字典: Key="small_portrait", Value=[paths...]
    
    # --- 准备统计 ---
    stats = {
        "total_rows_read": 0,
        "total_filtered_out_unsafe": 0,
        "total_filtered_out_parse_fail": 0,
        "standard_groups_count": 0,
        "standard_images_count": 0,
        "misc_images_count_by_bucket": defaultdict(int),
        "total_shard_files": 0
    }
    
    # 用于排序的缓冲区
    # 因为 SQL 没有 ORDER BY（无法直接用 h, w 排序），需要在内存中处理
    all_images = defaultdict(list)  # Key=(h,w), Value=[paths...]
    
    # 打开日志文件准备写入
    with open(REPORT_FILE, "w", encoding="utf-8") as f_log:
        
        def log(message, force_print=True):
            if force_print: print(message)
            f_log.write(message + "\n")

        log(f"Mode: {'DRY RUN (Preview Only)' if DRY_RUN else 'LIVE (WRITING FILES)'}")
        log(f"Output Dir: {OUTPUT_DIR if not DRY_RUN else 'N/A'}")
        log(f"Report File: {os.path.abspath(REPORT_FILE)}")
        log(f"Target Channel Count: 3")
        log("--- PATH EXCLUSIONS (Low Quality Filter) ---")
        for pattern in PATH_EXCLUSIONS:
            log(f"  - {pattern}")
        log("--- THRESHOLDS ---")
        log(f"  Safety Filter: < {MIN_PIXEL_EDGE}px or > {MAX_PIXEL_EDGE}px (DROPPED)")
        log(f"  Standard Group (Head): >= {MIN_GROUP_SIZE} images")
        log(f"  Misc Buckets (Tail): < {MIN_GROUP_SIZE} images, then bucketed by:")
        log(f"    - Size (max edge): {BUCKET_SIZE_THRESH}")
        log(f"    - AR (w/h): portrait <= {BUCKET_AR_THRESH['portrait']} | square < {BUCKET_AR_THRESH['square']} | landscape >= {BUCKET_AR_THRESH['square']}")
        log("-" * 60)

        try:
            log(f"Connecting to PostgreSQL at {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(name='task2_3ch_cursor')
            log("Executing SQL Query (Streaming)...")
            cursor.execute(get_sql_query())
        except psycopg2.OperationalError as e:
            log(f"❌ 数据库连接失败!")
            log(f"   错误信息: {e}")
            log(f"   请检查:")
            log(f"   1. PostgreSQL 服务是否运行: systemctl status postgresql 或 service postgresql status")
            log(f"   2. 数据库配置是否正确: host={DB_CONFIG['host']}, port={DB_CONFIG['port']}")
            log(f"   3. 可以通过环境变量设置: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
            log(f"   例如: export DB_HOST=your_host && export DB_PORT=5432")
            return
        except Exception as e:
            log(f"❌ 数据库查询失败: {e}")
            return

        # ============================================
        # PHASE 1: READ & CLASSIFY (In-Memory Buffer)
        # ============================================
        log("Phase 1: Reading from DB and parsing image shapes...")
        
        while True:
            rows = cursor.fetchmany(5000)
            if not rows:
                break
                
            for row in rows:
                stats["total_rows_read"] += 1
                if stats["total_rows_read"] % 1000000 == 0:
                    log(f"  ... read {stats['total_rows_read'] // 1000000}M rows", force_print=True)
                
                # row: (dataset_id, original_id, file_path, image_shape)
                d_id, _, f_path, image_shape = row
                
                # --- 1. 解析 image_shape ---
                h, w = parse_image_shape(image_shape)
                
                if h is None or w is None:
                    stats["total_filtered_out_parse_fail"] += 1
                    continue
                
                # --- 2. 安全过滤 (Safety Filter) ---
                if h > MAX_PIXEL_EDGE or w > MAX_PIXEL_EDGE or h < MIN_PIXEL_EDGE or w < MIN_PIXEL_EDGE:
                    stats["total_filtered_out_unsafe"] += 1
                    continue
                
                # --- 3. 按 (h, w) 分组 ---
                all_images[(h, w)].append(f_path)
        
        cursor.close()
        conn.close()
        log(f"Phase 1: DB Read Complete. Total groups: {len(all_images)}")

        # ============================================
        # PHASE 1.5: CLASSIFY GROUPS
        # ============================================
        log("Phase 1.5: Classifying groups into standard vs misc...")
        
        for (h, w), paths in all_images.items():
            count = len(paths)
            
            if count >= MIN_GROUP_SIZE:
                # 标准组
                standard_groups[(h, w)] = paths
                stats["standard_groups_count"] += 1
                stats["standard_images_count"] += count
            else:
                # 混合桶
                bucket_key = get_hybrid_bucket_key(h, w)
                misc_buckets[bucket_key].extend(paths)
                stats["misc_images_count_by_bucket"][bucket_key] += count
        
        # 释放内存
        del all_images
        log("Phase 1.5: Classification complete.")

        # ============================================
        # PHASE 2: WRITE SHARDS TO DISK
        # ============================================
        log(f"Phase 2: {'[Dry Run] Calculating shard statistics...' if DRY_RUN else 'Writing shards to disk...'}")
        
        # 2a. 写入标准组（按 h, w 排序以保持一致性）
        log(f"  ... processing {stats['standard_groups_count']} STANDARD groups...")
        sorted_standard_keys = sorted(standard_groups.keys())
        
        for (h, w) in sorted_standard_keys:
            paths = standard_groups[(h, w)]
            for i in range(0, len(paths), MAX_SHARD_SIZE):
                chunk = paths[i:i+MAX_SHARD_SIZE]
                filename = f"task_{stats['total_shard_files']:06d}_std_h{h}_w{w}.json"
                meta = {
                    "type": "standard",
                    "bucket": f"{h}x{w}",
                    "height": h,
                    "width": w,
                    "count": len(chunk)
                }
                write_shard_file(chunk, meta, filename)
                stats["total_shard_files"] += 1

        # 2b. 写入所有混合桶
        log(f"  ... processing {len(misc_buckets)} MISC buckets...")
        for bucket_key, paths in sorted(misc_buckets.items()):
            for i in range(0, len(paths), MAX_SHARD_SIZE):
                chunk = paths[i:i+MAX_SHARD_SIZE]
                filename = f"task_{stats['total_shard_files']:06d}_misc_{bucket_key}.json"
                meta = {
                    "type": "misc",
                    "bucket": bucket_key,
                    "height": -1, # -1 告诉 Ray Worker 需要动态 Resize
                    "width": -1,
                    "count": len(chunk)
                }
                write_shard_file(chunk, meta, filename)
                stats["total_shard_files"] += 1
            
        log("Phase 2: Complete.")
        
        # ============================================
        # PHASE 3: FINAL REPORT
        # ============================================
        end_time = time.time()
        log("\n" + "="*60)
        log(f"3-CHANNEL HYBRID SHARDING SUMMARY ({'DRY RUN' if DRY_RUN else 'LIVE'})")
        log("="*60)
        log(f"Total Time: {end_time - start_time:.2f} seconds")
        log(f"Total Rows Read from DB: {stats['total_rows_read']}")
        log(f"Images Filtered (Parse Fail): {stats['total_filtered_out_parse_fail']}")
        log(f"Images Filtered (Unsafe Size): {stats['total_filtered_out_unsafe']}")
        
        total_sharded = stats['standard_images_count'] + sum(stats['misc_images_count_by_bucket'].values())
        log(f"Total Images Sharded: {total_sharded}")
        log(f"Total JSON Files (Shards): {stats['total_shard_files']}")
        
        log("\n" + "-"*30)
        log(" STANDARD 'HEAD' GROUPS")
        log(f" (Groups with >= {MIN_GROUP_SIZE} images)")
        log(f"  - Total Standard Groups: {stats['standard_groups_count']}")
        log(f"  - Total Standard Images: {stats['standard_images_count']}")
        log(f"  - Total Standard Shards: {math.ceil(stats['standard_images_count'] / MAX_SHARD_SIZE) if stats['standard_images_count'] > 0 else 0}")
        
        log("\n" + "-"*30)
        log(" MISC 'TAIL' BUCKETS")
        log(f" (Groups with < {MIN_GROUP_SIZE} images)")
        log("-" * 60)
        log(f"{'Bucket Key':<20} | {'Image Count':<15} | {'Shard Count':<10}")
        log("-" * 60)
        
        sorted_misc_keys = sorted(misc_buckets.keys())
        total_misc_images = 0
        total_misc_shards = 0
        
        for key in sorted_misc_keys:
            count = stats['misc_images_count_by_bucket'][key]
            shards = math.ceil(count / MAX_SHARD_SIZE) if count > 0 else 0
            total_misc_images += count
            total_misc_shards += shards
            log(f"{key:<20} | {count:<15} | {shards:<10}")
            
        log("-" * 60)
        log(f"{'Total Misc':<20} | {total_misc_images:<15} | {total_misc_shards:<10}")
        log("="*60)

    print(f"\nDone! Report generated: {REPORT_FILE}")

if __name__ == "__main__":
    main()

