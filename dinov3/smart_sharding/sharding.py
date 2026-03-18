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
OUTPUT_DIR = "/mnt/huawei_deepcad/onepb/shards_task"
REPORT_FILE = "task2_hybrid_report.txt"  # 详细的统计报告

# --- 2. 核心参数 (Core Parameters) ---
MAX_SHARD_SIZE = 20000  # 每个 JSON 文件的最大图片数
EXCLUDE_PRIMARY = False

# 2a. 安全过滤 (直接丢弃)
MAX_PIXEL_EDGE = 5000  # 最长边 > 5k 
MIN_PIXEL_EDGE = 64     # 最短边 < 64

# 2b. "头部"与"尾部"的分割线
MIN_GROUP_SIZE = 100    # 分辨率组的图片数 >= 100，才算作“标准组”

# 2c. "尾部"分桶阈值
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

def get_sql_query():
    """构建核心 SQL，注意我们按 h, w 排序以实现分组"""
    base_query = """
    SELECT 
        0 as dataset_id,
        s.id as original_id, 
        s.file_path,
        CAST(SPLIT_PART(TRIM(BOTH '()' FROM s.image_shape), ',', 1) AS INTEGER) as h,
        CAST(SPLIT_PART(TRIM(BOTH '()' FROM s.image_shape), ',', 2) AS INTEGER) as w
    FROM original_images_all s
    WHERE s.channel_count = 1
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
        ORDER BY h, w  -- [重要] 必须排序才能在客户端进行分组
        """
    else:
        final_query = f"{base_query} ORDER BY h, w"
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
        "standard_groups_count": 0,
        "standard_images_count": 0,
        "misc_images_count_by_bucket": defaultdict(int),
        "total_shard_files": 0
    }
    
    # 打开日志文件准备写入
    with open(REPORT_FILE, "w", encoding="utf-8") as f_log:
        
        def log(message, force_print=True):
            if force_print: print(message)
            f_log.write(message + "\n")

        log(f"Mode: {'DRY RUN (Preview Only)' if DRY_RUN else 'LIVE (WRITING FILES)'}")
        log(f"Output Dir: {OUTPUT_DIR if not DRY_RUN else 'N/A'}")
        log(f"Report File: {os.path.abspath(REPORT_FILE)}")
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
            cursor = conn.cursor(name='task2_hybrid_cursor')
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
        log("Phase 1: Reading from DB and classifying all groups...")
        
        current_group_key = None
        current_group_buffer = []

        while True:
            rows = cursor.fetchmany(5000)
            if not rows:
                break
                
            for row in rows:
                stats["total_rows_read"] += 1
                if stats["total_rows_read"] % 1000000 == 0:
                    log(f"  ... read {stats['total_rows_read'] // 1000000}M rows", force_print=True)
                
                # row: (dataset_id, original_id, file_path, h, w)
                d_id, _, f_path, h, w = row
                
                # --- 1. 安全过滤 (Safety Filter) ---
                if h > MAX_PIXEL_EDGE or w > MAX_PIXEL_EDGE or h < MIN_PIXEL_EDGE or w < MIN_PIXEL_EDGE:
                    stats["total_filtered_out_unsafe"] += 1
                    continue
                
                group_key = (h, w)
                
                # --- 2. 组发生变化 (Group Change) ---
                if group_key != current_group_key:
                    if current_group_buffer:
                        # 处理上一个已完整的组
                        count = len(current_group_buffer)
                        prev_h, prev_w = current_group_key
                        
                        if count >= MIN_GROUP_SIZE:
                            # 2a. 归入标准组
                            standard_groups[(prev_h, prev_w)] = current_group_buffer
                            stats["standard_groups_count"] += 1
                            stats["standard_images_count"] += count
                        else:
                            # 2b. 归入混合桶
                            bucket_key = get_hybrid_bucket_key(prev_h, prev_w)
                            misc_buckets[bucket_key].extend(current_group_buffer)
                            stats["misc_images_count_by_bucket"][bucket_key] += count
                                
                    # 重置
                    current_group_buffer = [f_path] # 存入当前行
                    current_group_key = group_key
                else:
                    # 组未变，继续添加
                    current_group_buffer.append(f_path)
        
        # --- 3. 清理最后一个组 ---
        if current_group_buffer:
            count = len(current_group_buffer)
            prev_h, prev_w = current_group_key
            if count >= MIN_GROUP_SIZE:
                standard_groups[(prev_h, prev_w)] = current_group_buffer
                stats["standard_groups_count"] += 1
                stats["standard_images_count"] += count
            else:
                bucket_key = get_hybrid_bucket_key(prev_h, prev_w)
                misc_buckets[bucket_key].extend(current_group_buffer)
                stats["misc_images_count_by_bucket"][bucket_key] += count

        cursor.close()
        conn.close()
        log("Phase 1: DB Read Complete. All data buffered in memory.")

        # ============================================
        # PHASE 2: WRITE SHARDS TO DISK
        # ============================================
        log(f"Phase 2: {'[Dry Run] Calculating shard statistics...' if DRY_RUN else 'Writing shards to disk...'}")
        
        # 2a. 写入标准组
        log(f"  ... processing {stats['standard_groups_count']} STANDARD groups...")
        for (h, w), paths in standard_groups.items():
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
        for bucket_key, paths in misc_buckets.items():
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
        log(f"HYBRID SHARDING SUMMARY ({'DRY RUN' if DRY_RUN else 'LIVE'})")
        log("="*60)
        log(f"Total Time: {end_time - start_time:.2f} seconds")
        log(f"Total Rows Read from DB: {stats['total_rows_read']}")
        log(f"Images Filtered (Unsafe Size): {stats['total_filtered_out_unsafe']}")
        
        total_sharded = stats['standard_images_count'] + sum(stats['misc_images_count_by_bucket'].values())
        log(f"Total Images Sharded: {total_sharded}")
        log(f"Total JSON Files (Shards): {stats['total_shard_files']}")
        
        log("\n" + "-"*30)
        log(" STANDARD 'HEAD' GROUPS")
        log(f" (Groups with >= {MIN_GROUP_SIZE} images)")
        log(f"  - Total Standard Groups: {stats['standard_groups_count']}")
        log(f"  - Total Standard Images: {stats['standard_images_count']}")
        log(f"  - Total Standard Shards: {math.ceil(stats['standard_images_count'] / MAX_SHARD_SIZE)}")
        
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
            shards = math.ceil(count / MAX_SHARD_SIZE)
            total_misc_images += count
            total_misc_shards += shards
            log(f"{key:<20} | {count:<15} | {shards:<10}")
            
        log("-" * 60)
        log(f"{'Total Misc':<20} | {total_misc_images:<15} | {total_misc_shards:<10}")
        log("="*60)

    print(f"\nDone! Report generated: {REPORT_FILE}")

if __name__ == "__main__":
    main()
