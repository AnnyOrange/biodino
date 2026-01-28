import ray
import os
import socket
import sys

# ================= 核心修复区域开始 =================
# 1. 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 推导项目根目录 DINO_PATH
DINO_PATH = os.path.abspath(os.path.join(current_dir, "../../"))

# 3. 强制将根目录加入系统路径的最前面，防止 import dinov3 失败
if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)
    print(f"[System] Added {DINO_PATH} to PYTHONPATH")
# ================= 核心修复区域结束 =================

# Use a file path that we know exists on the driver (based on previous ls commands)
# This file was previously confirmed to exist on the driver machine
TEST_FILE = "/mnt/huawei_deepcad/onepb/shards_task2/misc/large_portrait/task_004349_misc_large_portrait.json"

print(f"Testing access to file: {TEST_FILE}")

# Connect to the Ray cluster
try:
    ray.init(address='auto')
except:
    print("Could not connect to existing cluster, starting local one...")
    ray.init()

@ray.remote(num_cpus=0.1, num_gpus=0.01)  # 强制调度到有 GPU 的 Worker 节点
def check_file(path):
    hostname = socket.gethostname()
    exists = os.path.exists(path)
    
    # Get actual IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
    except:
        ip_address = "unknown"

    try:
        # Try to list the directory to check permissions/mounting
        dir_path = os.path.dirname(path)
        if os.path.exists(dir_path):
            ls_out = str(os.listdir(dir_path)[:5]) # Just first 5 files
            dir_ok = True
            access_read = os.access(path, os.R_OK)
        else:
            ls_out = "Dir Not Found"
            dir_ok = False
            access_read = False
    except Exception as e:
        ls_out = str(e)
        dir_ok = False
        access_read = False
        
    return {
        "host": hostname,
        "ip": ip_address,
        "file_exists": exists,
        "dir_exists": dir_ok,
        "readable": access_read,
        "dir_content_sample": ls_out
    }

# Launch tasks across the cluster
# We launch more tasks than nodes to hopefully hit multiple different nodes
print("Submitting tasks to workers...")
try:
    resources = ray.cluster_resources()
    print(f"Cluster resources: {resources}")
    num_nodes = max(1, int(resources.get("node", 1)))
except:
    num_nodes = 1

# Try to distribute tasks across different nodes
# Launch multiple tasks per node to ensure we hit all nodes
tasks_per_node = 3
num_tasks = max(10, num_nodes * tasks_per_node)
print(f"Launching {num_tasks} tasks to distribute across {num_nodes} nodes...")

futures = [check_file.remote(TEST_FILE) for _ in range(num_tasks)]

print("Waiting for results...")
results = ray.get(futures)

# Aggregate results
print("\n=== Worker File Access Report ===")
results_by_host = {}
for res in results:
    host = res["host"]
    # Use host + ip as key to distinguish nodes with same hostname (unlikely but possible)
    key = f"{host} ({res['ip']})"
    if key not in results_by_host:
        results_by_host[key] = res

print(f"Found {len(results_by_host)} unique node(s) out of {num_nodes} total nodes in cluster.")
if len(results_by_host) == 1:
    print("⚠️  WARNING: Only one node returned results. This might mean:")
    print("   1. All tasks were scheduled to the same node (head node)")
    print("   2. Other nodes are not responding")
    print("   To verify, you may need to SSH to other nodes and check if they can access the file.")
    print()

for host_info, res in results_by_host.items():
    status = "✅ OK" if res["file_exists"] and res["readable"] else "❌ FAIL"
    print(f"Node: {host_info:<30} | Status: {status}")
    print(f"  File Exists: {res['file_exists']}")
    print(f"  Readable:    {res['readable']}")
    if not res['file_exists']:
        print(f"  Dir Access:  {res['dir_content_sample']}")
    print("-" * 40)

# Summary
all_ok = all(r["file_exists"] and r["readable"] for r in results_by_host.values())
if all_ok:
    print("\n✅ All tested nodes can access the file.")
else:
    failed_nodes = [k for k, v in results_by_host.items() if not (v["file_exists"] and v["readable"])]
    print(f"\n❌ {len(failed_nodes)} node(s) cannot access the file: {failed_nodes}")

