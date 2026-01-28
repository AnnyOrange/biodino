import ray
import torch
import numpy as np
import os
import json
import gc
from typing import List, Dict, Any
from processor import ImageProcessor
import sys
from concurrent.futures import ThreadPoolExecutor

# Optimize CUDA allocation for tight memory (24GB cards loading 7B model)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@ray.remote(num_gpus=1)
class DINOv3Actor:
    def __init__(self, model_name='vit_7b', checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor_cache = {} 
        
        print(f"Initializing DINOv3Actor on {self.device}")
        
        # Pre-check: Check initial GPU Memory
        if torch.cuda.is_available():
            # Force clear any pre-allocated cache
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem_gb = free_mem / (1024**3)
            total_mem_gb = total_mem / (1024**3)
            print(f"GPU Memory: {free_mem_gb:.2f}GB free / {total_mem_gb:.2f}GB total")
            
            if total_mem_gb > 35.0: # A100 case
                # Require < 20% usage (i.e., > 80% free)
                required_mem = total_mem_gb * 0.8
                if free_mem_gb < required_mem:
                    err_msg = f"A100 usage too high (>20%). Need {required_mem:.1f}GB free, found {free_mem_gb:.2f}GB. Skipping."
                    print(err_msg)
                    raise RuntimeError(err_msg)
            else: # 3090 case
                if free_mem_gb < 15.0:
                    err_msg = f"Insufficient GPU memory. Need 15GB, found {free_mem_gb:.2f}GB. Skipping this worker."
                    print(err_msg)
                    raise RuntimeError(err_msg)

        # Load Model
        try:
            # Import inside the actor to ensure environment is ready
            from dinov3.models import vision_transformer as vits
            
            # 1. Create Model on CPU (FP32, ~26GB RAM)
            # Config matched from checkpoint: SwiGLU, No QKV bias, Has FFN/Proj bias, Has LayerScale
            self.model = vits.vit_7b(
                img_size=512, 
                qkv_bias=False, 
                ffn_layer="swiglu",
                ffn_bias=True,   
                proj_bias=True,
                layerscale_init=1e-4 
            )
            
            # 2. CRITICAL FIX: Convert to BF16/FP16 on CPU *BEFORE* moving to GPU
            # This reduces size from ~26GB to ~13GB, fitting into 3090's 24GB.
            if torch.cuda.is_bf16_supported():
                print("Converting to bfloat16 (CPU)...")
                self.model.bfloat16()
                self.dtype = torch.bfloat16
            else:
                print("Converting to float16 (CPU)...")
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
                
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f"Weights loaded: {msg}")
                del state_dict
                gc.collect()
            else:
                print("WARNING: No checkpoint loaded. Using random initialization.")

            # 4. Move to GPU (Now it is only ~13GB)
            print("Moving model to GPU...")
            self.model.to(self.device)
            self.model.eval()
            
            # Post-load Memory Check & Auto-Scaling Batch Size
            if torch.cuda.is_available():
                free_mem_after, _ = torch.cuda.mem_get_info()
                self.free_mem_gb_runtime = free_mem_after / (1024**3)
                print(f"Memory after model load: {self.free_mem_gb_runtime:.2f}GB free")
                
                # Auto-Scaling (Coefficients verified by demo stress test)
                # 512x512: 0.4GB/img (Real ~0.12)
                # 256x256: 0.12GB/img (Real ~0.03)
                safe_mem = max(0.5, self.free_mem_gb_runtime - 2.0)
                self.bs_512_cached = int(safe_mem / 0.4) 
                self.bs_256_cached = int(safe_mem / 0.12)
                
                self.bs_512_cached = max(1, min(self.bs_512_cached, 128)) 
                self.bs_256_cached = max(4, min(self.bs_256_cached, 256))
                print(f"Initial Batch Config: 512->{self.bs_512_cached}, 256->{self.bs_256_cached}")
            else:
                self.free_mem_gb_runtime = 0
                self.bs_512_cached = 1
                self.bs_256_cached = 4
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def ping(self):
        return "pong"

    def extract_shard(self, json_path: str, output_dir: str, ar_strategy: str):
        try:
            shard_name = os.path.splitext(os.path.basename(json_path))[0]
            shard_id = shard_name
            
            processor = ImageProcessor(ar_strategy=ar_strategy)
            
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Improved JSON Parsing Logic
            items = []
            global_meta = {}
            
            if isinstance(data, dict):
                # Extract global meta if present
                if 'meta' in data and isinstance(data['meta'], dict):
                    global_meta = data['meta']
                
                # Determine list key
                list_key = None
                if 'images' in data: list_key = 'images'
                elif 'files' in data: list_key = 'files'
                
                if list_key:
                    # Case 1: {"meta": ..., "files": ["path1", "path2"]} or [{"path":...}]
                    raw_list = data[list_key]
                    for entry in raw_list:
                        if isinstance(entry, str):
                            # Entry is just a path string
                            items.append({'path': entry, 'meta': global_meta})
                        elif isinstance(entry, dict):
                            # Entry is a dict
                            # Ensure 'meta' exists
                            if 'meta' not in entry:
                                entry['meta'] = global_meta
                            # Ensure 'path' exists (handle variations if needed)
                            items.append(entry)
                else:
                    # Case 2: {"path1": {meta}, "path2": {meta}}
                    # Must skip 'meta' key if it exists at top level
                    for k, v in data.items():
                        if k == 'meta': continue
                        
                        item = {'path': k}
                        if isinstance(v, dict):
                            item.update(v)
                            # Merge global meta if local is missing
                            if 'meta' not in item:
                                item['meta'] = global_meta
                        items.append(item)
            elif isinstance(data, list):
                # Case 3: [{"path":...}, ...]
                items = data
            else:
                print(f"Unknown JSON structure in {json_path}")
                return 0

            feature_list = []
            valid_paths = []
            
            BATCH_SIZE_MAP = {
                256: self.bs_256_cached,
                512: self.bs_512_cached
            }
            DEFAULT_BATCH_SIZE = self.bs_256_cached // 2
            
            current_batch_tensors = []
            current_batch_paths = []
            current_h, current_w = -1, -1

            with torch.no_grad():
                for item in items:
                    path = item.get('path') or item.get('image_path')
                    if not path: continue
                    
                    if not os.path.exists(path):
                        continue

                    type_label = item.get('meta', {}).get('type_label', 'standard_small_square')

                    try:
                        processed = processor.process(path, type_label)
                    except Exception as e:
                        print(f"Skipping corrupt image {path}: {e}")
                        continue

                    if isinstance(processed, list):
                        # Flush existing batch
                        if current_batch_tensors:
                            self._flush_batch(current_batch_tensors, current_batch_paths, feature_list, valid_paths)
                            current_batch_tensors = []
                            current_batch_paths = []
                            current_h, current_w = -1, -1

                        # Process patches (convert dtype)
                        # processor output is float32, model is bf16/half
                        patches = torch.cat(processed, dim=0).to(self.device, dtype=self.dtype)
                        
                        HUGE_BATCH_LIMIT = self.bs_512_cached
                        
                        patch_feats = []
                        for i in range(0, patches.shape[0], HUGE_BATCH_LIMIT):
                            batch_patches = patches[i:i+HUGE_BATCH_LIMIT]
                            batch_out = self.model(batch_patches)
                            patch_feats.append(batch_out)
                        
                        if patch_feats:
                            feats = torch.cat(patch_feats, dim=0) # (N, 4096)
                            img_feat = feats.mean(dim=0, keepdim=True) # (1, 4096)
                            
                            # Fix: Convert BF16 -> Float32 -> Numpy
                            feature_list.append(img_feat.float().cpu().numpy())
                            valid_paths.append(path)
                        
                    else:
                        # Standard image
                        tensor = processed
                        _, _, h, w = tensor.shape
                        target_batch_size = BATCH_SIZE_MAP.get(h, DEFAULT_BATCH_SIZE)

                        if (current_h != -1 and (h != current_h or w != current_w)):
                            self._flush_batch(current_batch_tensors, current_batch_paths, feature_list, valid_paths)
                            current_batch_tensors = []
                            current_batch_paths = []
                        
                        current_h, current_w = h, w
                        current_batch_tensors.append(tensor)
                        current_batch_paths.append(path)
                        
                        if len(current_batch_tensors) >= target_batch_size:
                            self._flush_batch(current_batch_tensors, current_batch_paths, feature_list, valid_paths)
                            current_batch_tensors = []
                            current_batch_paths = []
                            current_h, current_w = -1, -1

                if current_batch_tensors:
                    self._flush_batch(current_batch_tensors, current_batch_paths, feature_list, valid_paths)

            if feature_list:
                all_features = np.concatenate(feature_list, axis=0).astype(np.float16)
                
                os.makedirs(output_dir, exist_ok=True)
                npy_path = os.path.join(output_dir, f"features_{shard_id}.npy")
                txt_path = os.path.join(output_dir, f"valid_paths_{shard_id}.txt")
                
                np.save(npy_path, all_features)
                with open(txt_path, 'w') as f:
                    for p in valid_paths:
                        f.write(f"{p}\n")
                
                return len(valid_paths)
            else:
                return 0

        except Exception as e:
            print(f"Failed to process shard {json_path}: {e}")
            raise e

    def _flush_batch(self, tensors, paths, feature_list, valid_paths):
        if not tensors:
            return
        
        # Explicit cast to self.dtype (bf16 or half) to avoid implicit FP32 cast
        batch = torch.cat(tensors, dim=0).to(self.device, dtype=self.dtype)
        feats = self.model(batch) # (B, 4096)
        
        # Fix: Convert BF16 -> Float32 -> Numpy (Numpy doesn't support BF16)
        feature_list.append(feats.float().cpu().numpy())
        valid_paths.extend(paths)
