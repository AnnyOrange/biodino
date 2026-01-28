import argparse
import os
import sys
import glob
import ray
import numpy as np
import time

# ================= 环境路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设结构是 root/dinov3/feature_extraction/main.py -> root
DINO_PATH = os.path.abspath(os.path.join(current_dir, "../../"))

if DINO_PATH not in sys.path:
    sys.path.insert(0, DINO_PATH)
    print(f"[System] Added {DINO_PATH} to PYTHONPATH")

# 尝试导入
try:
    from dinov3.feature_extraction.actors_v2 import DINOv3Actor
except ImportError:
    import sys
    sys.path.append(current_dir)
    from actors_v2 import DINOv3Actor
# ===============================================

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from ray.util.actor_pool import ActorPool

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed DINOv3 Feature Extraction")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing JSON shards")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save output NPY files")
    parser.add_argument("--ar_strategy", type=str, choices=['crop', 'warp', 'tile'], default='crop',
                        help="Strategy for handling non-square aspect ratios")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to DINOv3 weights")
    return parser.parse_args()

def get_output_filename(json_path):
    # 逻辑必须与 actors_v2.py 中的 shard_id 生成逻辑一致
    shard_name = os.path.splitext(os.path.basename(json_path))[0]
    return f"features_{shard_name}.npy"

def main():
    args = parse_args()
    console = Console()

    # 1. 初始化 Ray
    runtime_env = {
        "working_dir": DINO_PATH,
        "env_vars": {
            "PYTHONPATH": f"{DINO_PATH}:{current_dir}:{os.getenv('PYTHONPATH', '')}",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        },
        "excludes": ["*.pth", "*.npy", "output/", "largedata_npy/", ".git/"] 
    }

    console.print(f"[bold green]Initializing Ray Cluster...[/bold green]")
    ray.init(address='auto', runtime_env=runtime_env, ignore_reinit_error=True)
    
    resources = ray.cluster_resources()
    num_gpus = int(resources.get("GPU", 0))
    available_cpus = int(resources.get("CPU", 0))
    console.print(f"Cluster Resources: {resources}")
    console.print(f"Detected [bold blue]{num_gpus} GPUs[/bold blue].")
    console.print(f"Detected [bold blue]{available_cpus} CPUs[/bold blue].")

    if available_cpus <= 0:
        console.print("[bold red]Ray reports 0 CPUs. Unable to launch actors.[/bold red]")
        return

    # 2. 扫描任务 & 断点续传过滤
    console.print(f"Scanning input directory: {args.input_dir}...")
    all_json_files = glob.glob(os.path.join(args.input_dir, "**/*.json"), recursive=True)
    
    if not all_json_files:
        console.print("[red]No JSON files found![/red]")
        return

    console.print(f"Checking existing outputs in: {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    tasks_to_run = []
    skipped_count = 0
    
    # 预先获取输出目录下的所有文件名，避免循环中反复 IO
    existing_files = set(os.listdir(args.output_dir))

    for f in all_json_files:
        target_name = get_output_filename(f)
        if target_name in existing_files:
            skipped_count += 1
        else:
            tasks_to_run.append(f)

    console.print(f"Total Shards: {len(all_json_files)}")
    console.print(f"Skipped: [yellow]{skipped_count}[/yellow] (Already processed)")
    console.print(f"Remaining: [bold green]{len(tasks_to_run)}[/bold green]")

    if not tasks_to_run:
        console.print("[bold green]All tasks completed! Nothing to do.[/bold green]")
        return

    # 3. 启动 Actors
    desired_actor_count = num_gpus if num_gpus > 0 else 1
    cpu_options = [10, 6, 4]
    num_cpus_per_actor = cpu_options[0]
    actor_count = desired_actor_count

    cpu_option_found = False
    for option in cpu_options:
        if option * desired_actor_count <= available_cpus:
            num_cpus_per_actor = option
            cpu_option_found = True
            break

    if not cpu_option_found:
        viable_options = [opt for opt in cpu_options if 0 < opt <= available_cpus]
        if viable_options:
            num_cpus_per_actor = min(viable_options)
        else:
            num_cpus_per_actor = max(1, available_cpus)

        max_supported = available_cpus // max(1, num_cpus_per_actor)
        if max_supported == 0:
            actor_count = 1
            num_cpus_per_actor = max(1, min(num_cpus_per_actor, available_cpus))
        else:
            actor_count = max(1, min(desired_actor_count, max_supported))

        if actor_count < desired_actor_count:
            console.print(f"[yellow]CPU constrained: reducing actor count from {desired_actor_count} to {actor_count}.[/yellow]")
    elif num_cpus_per_actor != cpu_options[0]:
        console.print(f"[yellow]CPU constrained: using {num_cpus_per_actor} CPUs per actor instead of {cpu_options[0]}.[/yellow]")

    total_cpu_needed = actor_count * max(1, num_cpus_per_actor)
    if total_cpu_needed > available_cpus:
        actor_count = max(1, available_cpus // max(1, num_cpus_per_actor))
        total_cpu_needed = actor_count * max(1, num_cpus_per_actor)

    console.print(f"Initializing {actor_count} Actors (each {num_cpus_per_actor} CPUs)...")

    # 批量创建 Actor handle
    actor_handles = [
        DINOv3Actor.options(num_cpus=num_cpus_per_actor).remote(
            checkpoint_path=args.checkpoint_path if args.checkpoint_path else None
        )
        for _ in range(actor_count)
    ]
    
    healthy_actors = []
    # 并行 Ping 检测，设置较长的 timeout（400秒），给节点足够的模型加载和初始化时间
    ping_refs = [actor.ping.remote() for actor in actor_handles]
    
    try:
        # 使用 ray.wait 动态获取就绪的 actor，给 400秒 超时
        start_time = time.time()
        ready, not_ready = ray.wait(ping_refs, num_returns=len(ping_refs), timeout=700)
        
        # 建立 ref 到 actor 的映射
        ref_to_actor = {ref: actor for ref, actor in zip(ping_refs, actor_handles)}
        
        for ref in ready:
            try:
                ray.get(ref) # 确认没有抛出异常
                healthy_actors.append(ref_to_actor[ref])
            except Exception as e:
                print(f"[WARN] An actor failed to initialize: {e}")
        
        # ========== 限制每个 A100 节点最多 4 张卡 ==========
        console.print("[yellow]Checking GPU types and limiting A100 nodes to 4 GPUs...[/yellow]")
        
        # 查询所有健康 Actor 的节点信息
        node_info_refs = [actor.get_node_info.remote() for actor in healthy_actors]
        node_infos = []
        
        for i, info_ref in enumerate(node_info_refs):
            try:
                info = ray.get(info_ref, timeout=10)
                node_infos.append((i, info))
            except Exception as e:
                print(f"[WARN] Failed to get node info for actor {i}: {e}")
                # 如果无法获取节点信息，暂时保留该 Actor（添加到其他节点列表中）
                node_infos.append((i, {"hostname": "unknown", "is_a100": False, "gpu_total_mem_gb": 0}))
        
        # 统计每个 A100 节点的 Actor 数量
        a100_node_counts = {}  # {hostname: [actor_indices]}
        other_actors = []  # 非 A100 节点的 Actor
        
        for actor_idx, info in node_infos:
            if info["is_a100"]:
                hostname = info["hostname"]
                if hostname not in a100_node_counts:
                    a100_node_counts[hostname] = []
                a100_node_counts[hostname].append(actor_idx)
            else:
                other_actors.append(actor_idx)
        
        # 限制每个 A100 节点最多 3 个 Actor（降低以缓解 CPU 内存压力）
        A100_MAX_ACTORS_PER_NODE = 4
        final_actors = []
        
        for hostname, actor_indices in a100_node_counts.items():
            if len(actor_indices) > A100_MAX_ACTORS_PER_NODE:
                console.print(f"[yellow]A100 node {hostname}: {len(actor_indices)} actors -> limiting to {A100_MAX_ACTORS_PER_NODE}[/yellow]")
                # 保留前 4 个
                selected = actor_indices[:A100_MAX_ACTORS_PER_NODE]
                final_actors.extend(selected)
                # 删除多余的 Actor
                for idx in actor_indices[A100_MAX_ACTORS_PER_NODE:]:
                    try:
                        ray.kill(healthy_actors[idx])
                    except:
                        pass
            else:
                final_actors.extend(actor_indices)
        
        # 添加所有非 A100 节点的 Actor
        final_actors.extend(other_actors)
        
        # 更新 healthy_actors 列表
        healthy_actors = [healthy_actors[i] for i in final_actors]
        
        console.print(f"[green]After A100 limiting: {len(healthy_actors)} actors[/green]")
        # ==================================================
                
        console.print(f"Healthy Actors: [bold green]{len(healthy_actors)}[/bold green] / {actor_count}")
        if not_ready:
            console.print(f"[yellow]Warning: {len(not_ready)} actors timed out during initialization and will be ignored.[/yellow]")

    except Exception as e:
        console.print(f"[red]Critical error during actor init: {e}[/red]")

    if not healthy_actors:
        console.print("[bold red]No healthy actors available! Exiting.[/bold red]")
        return

    pool = ActorPool(healthy_actors)

    # 4. 执行处理
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task_id = progress.add_task("Processing", total=len(tasks_to_run))
        
        def process_shard(actor, shard_path):
            return actor.extract_shard.remote(shard_path, args.output_dir, args.ar_strategy)

        # 用于记录失败的 shard，以便后续重试
        failed_shards = []
        
        try:
            for result in pool.map_unordered(process_shard, tasks_to_run):
                progress.advance(task_id)
                
                if result is None:
                    print("[WARN] A task returned None (Worker crashed?)")
                    continue
                
                try:
                    # 解析返回值 (shard_id, count)
                    if isinstance(result, tuple):
                        shard_id, count = result
                        if count == 0:
                            print(f"[WARN] Zero features extracted for: {shard_id}")
                    elif isinstance(result, int):
                        pass 

                except Exception as e:
                     print(f"[ERROR] Result parsing failed: {e}")
                     
        except ray.exceptions.ActorUnavailableError as e:
            console.print(f"[red]Actor unavailable error occurred: {e}[/red]")
            console.print("[yellow]This usually means an Actor crashed or the node went down.[/yellow]")
            console.print("[yellow]The program will continue with remaining Actors.[/yellow]")
            # 可以在这里记录失败的 shard，但不中断整个流程
            # 由于 ActorPool 的内部实现，我们无法直接获取当前失败的 shard
            # 但剩余的 Actor 会继续处理其他任务
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted by user. Exiting...[/yellow]")
            raise
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            raise

    console.print("[bold green]Processing Complete![/bold green]")

if __name__ == "__main__":
    main()
