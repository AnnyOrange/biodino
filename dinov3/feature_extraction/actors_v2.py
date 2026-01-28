import ray
import torch
import numpy as np
import os
import json
import gc
import socket
import time
import random
from typing import List, Dict, Any
import sys

from dinov3.feature_extraction.processor import ImageProcessor
from concurrent.futures import ThreadPoolExecutor

# 启用 PyTorch 显存碎片整理优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@ray.remote(num_gpus=1, num_cpus=10)
class DINOv3Actor:
    def __init__(self, model_name='vit_7b', checkpoint_path=None):
        # [FIX] 随机睡眠已移除。上次引入是为了防止 CPU OOM，但现在是导致初始化超时的原因。
        # time.sleep(random.uniform(5, 45))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hostname = socket.gethostname()
        print(f"Initializing DINOv3Actor on {self.device} (Host: {self.hostname})")
        
        # 显存预检查
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem_gb = free_mem / (1024**3)
            total_mem_gb = total_mem / (1024**3)
            print(f"GPU Memory: {free_mem_gb:.2f}GB free / {total_mem_gb:.2f}GB total")
            
            if total_mem_gb > 35.0: # A100 (40G/80G)
                required_mem = total_mem_gb * 0.7
                if free_mem_gb < required_mem:
                    raise RuntimeError(f"A100 usage too high. Need {required_mem:.1f}GB, found {free_mem_gb:.2f}GB.")
            else: # 3090 / V100
                if free_mem_gb < 14.0:
                    raise RuntimeError(f"Insufficient GPU memory. Need 15GB, found {free_mem_gb:.2f}GB.")

        # 加载模型
        try:
            from dinov3.models import vision_transformer as vits
            
            self.model = vits.vit_7b(
                img_size=512, 
                qkv_bias=False, 
                ffn_layer="swiglu",
                ffn_bias=True,   
                proj_bias=True,
                layerscale_init=1e-4 
            )
            
            if torch.cuda.is_bf16_supported():
                print("Using bfloat16")
                self.model.bfloat16()
                self.dtype = torch.bfloat16
            else:
                print("Using float16")
                self.model.half()
                self.dtype = torch.float16
            
            # 3. Load Weights (into CPU BF16 model)
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                default_ckpt = "/mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
                if os.path.exists(default_ckpt):
                    checkpoint_path = default_ckpt
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading weights from {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                if 'teacher' in state_dict: state_dict = state_dict['teacher']
                elif 'model' in state_dict: state_dict = state_dict['model']
                self.model.load_state_dict(state_dict, strict=False)
                del state_dict
                gc.collect()
            else:
                print("WARNING: Random initialization (No checkpoint found).")

            self.model.to(self.device)
            self.model.eval()
            
            # 动态设定 Batch Size
            if torch.cuda.is_available():
                free_mem_after, _ = torch.cuda.mem_get_info()
                self.free_mem_gb_runtime = free_mem_after / (1024**3)
                
                safe_mem = max(0.5, self.free_mem_gb_runtime - 4.0) 
                self.bs_512_cached = int(safe_mem / 0.4) 
                self.bs_256_cached = int(safe_mem / 0.12)
                
                # 限制最大值
                self.bs_512_cached = max(1, min(self.bs_512_cached, 64)) 
                self.bs_256_cached = max(4, min(self.bs_256_cached, 196))
                print(f"Batch Config: 512->{self.bs_512_cached}, 256->{self.bs_256_cached}")
            else:
                self.free_mem_gb_runtime = 0
                self.bs_512_cached = 1
                self.bs_256_cached = 4
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def ping(self):
        return "pong"

    def get_node_info(self):
        """返回节点信息：hostname 和是否是 A100 节点"""
        import torch
        is_a100 = False
        total_mem_gb = 0
        if torch.cuda.is_available():
            _, total_mem = torch.cuda.mem_get_info()
            total_mem_gb = total_mem / (1024**3)
            # A100 通常是 40GB 或 80GB
            is_a100 = total_mem_gb >= 38.0  # 阈值设为 38GB 以覆盖 40GB A100
        return {
            "hostname": self.hostname,
            "is_a100": is_a100,
            "gpu_total_mem_gb": total_mem_gb
        }

    def extract_shard(self, json_path: str, output_dir: str, ar_strategy: str):
        # 获取当前 hostname 以便调试
        hostname = self.hostname
        
        try:
            shard_name = os.path.splitext(os.path.basename(json_path))[0]
            shard_id = shard_name
            processor = ImageProcessor(ar_strategy=ar_strategy)
            
            # 检查 JSON 是否存在
            if not os.path.exists(json_path):
                parent_dir = os.path.dirname(json_path)
                parent_exists = os.path.exists(parent_dir)
                
                # 详细的错误诊断
                print(f"[ERR] JSON not found on {hostname}: {json_path}")
                if not parent_exists:
                    print(f"[ERR]   Parent directory also missing: {parent_dir}")
                    # 检查 NFS 挂载点
                    if parent_dir.startswith('/mnt/huawei_deepcad'):
                        mount_exists = os.path.exists('/mnt/huawei_deepcad')
                        print(f"[ERR]   NFS mount /mnt/huawei_deepcad exists: {mount_exists}")
                        if not mount_exists:
                            print(f"[ERR]   ⚠️  NFS NOT MOUNTED on {hostname}! Check mount configuration.")
                else:
                    print(f"[ERR]   Parent dir exists but file missing - may be NFS sync issue or file deleted")
                return (shard_id, 0)

            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # ================= JSON 解析适配 =================
            # 1. 获取全局 Meta (这是关键，您的 type_label 在这里)
            global_meta = data.get('meta', {}) if isinstance(data, dict) else {}
            # 默认 fallback
            global_type_label = global_meta.get('type_label', 'standard_small_square')
            
            print(f"[DEBUG] Worker {hostname} processing {shard_id}. Global label: {global_type_label}")

            # 2. 提取文件列表
            items = []
            if isinstance(data, dict):
                if 'images' in data: 
                    items = data['images']
                elif 'files' in data:  # <--- 适配您的 JSON 结构
                    items = data['files']
                else:
                    # 兜底：如果是个字典但没找到 list，可能 key 就是路径
                    for k, v in data.items():
                        if k == 'meta': continue # 跳过 meta key
                        if isinstance(v, dict):
                            items.append({'path': k, **v})
                        else:
                            items.append({'path': k})
            elif isinstance(data, list):
                items = data
            
            if not items:
                print(f"[WARN] Worker {hostname}: No items found in {shard_id}")
                return (shard_id, 0)

            # ... 准备批处理变量 ...
            feature_list = []
            valid_paths = []
            BATCH_SIZE_MAP = {256: self.bs_256_cached, 512: self.bs_512_cached}
            DEFAULT_BATCH_SIZE = self.bs_256_cached // 2
            current_batch_tensors = []
            current_batch_paths = []
            current_h, current_w = -1, -1

            # ==================================================
            # [FIX] 增加文件路径检查日志
            # 如果文件不存在，打印错误（抽样打印防止刷屏）
            # ==================================================
            def load_and_process(item):
                if isinstance(item, str): path = item
                else: path = item.get('path') or item.get('image_path')

                if not path: return None

                if not os.path.exists(path):
                    # [DEBUG] 关键调试信息
                    if random.random() < 0.001: # 千分之一概率打印，证明有节点找不到文件
                        print(f"[ERR] File missing on {hostname}: {path}")
                    return None
                
                try:
                    item_meta = item.get('meta', {}) if isinstance(item, dict) else {}
                    type_label = item_meta.get('type_label') or item.get('type_label') or global_type_label
                    processed = processor.process(path, type_label)
                    return (path, processed)
                except Exception as e:
                    if random.random() < 0.001:
                        print(f"[WARN] Image process failed on {hostname}: {path} - {e}")
                    return None

            # [FIX] 降低 CPU 并发，防止内存溢出
            NUM_WORKERS = 4 
            with torch.no_grad():
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    results = executor.map(load_and_process, items)
                    
                    for res in results:
                        if res is None: continue
                        path, processed = res
                        
                        if isinstance(processed, list):
                            # Huge Image / Tile Logic
                            if current_batch_tensors:
                                self._flush_batch(current_batch_tensors, current_batch_paths, feature_list, valid_paths)
                                current_batch_tensors = []
                                current_batch_paths = []
                                current_h, current_w = -1, -1
                            
                            # [Note] 如果不需要降级保护，这里依然加上基本的 try-except 以防万一
                            # 但不会主动降低 Batch Size
                            try:
                                patches = torch.cat(processed, dim=0).to(self.device)
                                if self.dtype == torch.bfloat16: patches = patches.bfloat16()
                                else: patches = patches.half()
                                
                                BS = self.bs_512_cached
                                patch_feats = []
                                for i in range(0, patches.shape[0], BS):
                                    batch_patches = patches[i:i+BS]
                                    batch_out = self.model(batch_patches)
                                    patch_feats.append(batch_out)
                                
                                if patch_feats:
                                    feats = torch.cat(patch_feats, dim=0)
                                    img_feat = feats.mean(dim=0, keepdim=True)
                                    feature_list.append(img_feat.float().cpu().numpy())
                                    valid_paths.append(path)
                            except Exception as e:
                                print(f"[ERROR] GPU Forward failed for huge image {path}: {e}")
                                torch.cuda.empty_cache()
                                continue

                        else:
                            # Standard Image 处理逻辑
                            tensor = processed
                            _, _, h, w = tensor.shape
                            target_batch_size = BATCH_SIZE_MAP.get(h, DEFAULT_BATCH_SIZE)

                            if (current_h != -1 and (h != current_h or w != current_w)) or \
                               (len(current_batch_tensors) >= target_batch_size):
                                self._flush_batch(current_batch_tensors, current_batch_paths, feature_list, valid_paths)
                                current_batch_tensors = []
                                current_batch_paths = []
                            
                            current_h, current_w = h, w
                            current_batch_tensors.append(tensor)
                            current_batch_paths.append(path)

                if current_batch_tensors:
                    self._flush_batch(current_batch_tensors, current_batch_paths, feature_list, valid_paths)

            # ================= Worker 端直接保存结果 =================
            if feature_list:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 压缩为 float16 节省空间
                    all_features = np.concatenate(feature_list, axis=0).astype(np.float16)
                    
                    npy_path = os.path.join(output_dir, f"features_{shard_id}.npy")
                    txt_path = os.path.join(output_dir, f"valid_paths_{shard_id}.txt")
                    
                    np.save(npy_path, all_features)
                    with open(txt_path, 'w') as f:
                        for p in valid_paths:
                            f.write(f"{p}\n")
                    
                    # 随机清理缓存
                    if random.random() < 0.1: 
                        torch.cuda.empty_cache()
                        
                    return (shard_id, len(valid_paths))

                except Exception as e:
                    print(f"[ERROR] Save failed for {shard_id}: {e}")
                    raise e
            else:
                # [DEBUG] 明确指出是哪个节点没有产出
                print(f"[WARN] {hostname}: Processed {len(items)} items but produced 0 features for {shard_id} (Check file paths!)")
                return (shard_id, 0)

        except Exception as e:
            print(f"Failed to process shard {json_path}: {e}")
            # 返回错误标识
            return ("error", 0)

    def _flush_batch(self, tensors, paths, feature_list, valid_paths):
        if not tensors: return
        try:
            batch = torch.cat(tensors, dim=0).to(self.device)
            if self.dtype == torch.bfloat16: batch = batch.bfloat16()
            else: batch = batch.half()
                
            feats = self.model(batch)
            feature_list.append(feats.float().cpu().numpy())
            valid_paths.extend(paths)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[OOM] Batch flush failed on {self.hostname}. Cleaning cache.")
                torch.cuda.empty_cache()
            else:
                print(f"[ERROR] Batch flush failed: {e}")

