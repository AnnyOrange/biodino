import torch
import os
import gc
import time
import sys

# Set env var before anything else
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def print_mem(tag=""):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_gb = total_gb - free_gb
        print(f"[{tag}] Memory: Used {used_gb:.2f}GB / Free {free_gb:.2f}GB (Total {total_gb:.2f}GB)")
    else:
        print(f"[{tag}] CUDA not available")

def demo():
    print("Starting Single GPU Demo (Stress Test)...")
    print_mem("Start")
    
    # Add path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(current_dir)
    repo_root = os.path.dirname(package_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        
    try:
        from dinov3.models import vision_transformer as vits
    except ImportError as e:
        print(f"Import failed: {e}")
        if os.path.exists("dinov3"):
            sys.path.append(os.getcwd())
            from dinov3.models import vision_transformer as vits

    device = torch.device("cuda")
    
    print("Creating Model (CPU, FP32)...")
    # Matches actors.py config (SwiGLU, No Bias, LayerScale)
    model = vits.vit_7b(
        img_size=512, 
        qkv_bias=False, 
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        layerscale_init=1e-4
    )
    print_mem("After Init CPU")
    
    print("Converting to BF16 (CPU)...")
    if torch.cuda.is_bf16_supported():
        model.bfloat16()
        dtype = torch.bfloat16
        print("Using bfloat16")
    else:
        model.half()
        dtype = torch.float16
        print("Using float16")
        
    print_mem("After Half CPU")
    
    print("Moving to CUDA...")
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        model.to(device)
    except RuntimeError as e:
        print(f"OOM during .to(device): {e}")
        return

    print_mem("After .to(device)")
    
    model.eval()
    
    # Load Weights (Skipping for speed if verified, but good to be realistic)
    ckpt_path = "/mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "model" in state_dict: state_dict = state_dict["model"]
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Weights Loaded: {msg}")
    else:
        print("Checkpoint not found, skipping load.")
        
    print_mem("After Load Weights")
    
    # Stress Test 1: 256x256 Batch 32
    try:
        bs = 32
        print(f"\n[Test 1] 256x256 Batch={bs}")
        x = torch.randn(bs, 3, 256, 256, device=device, dtype=dtype)
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        with torch.no_grad():
            out = model(x)
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated()
        print(f"Output shape: {out.shape}")
        print(f"Inference Delta: {(end_mem - start_mem)/1024**3:.4f} GB")
        print_mem(f"After Batch {bs}")
        del x, out
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"[Test 1] Failed OOM: {e}")

    # Stress Test 2: 256x256 Batch 64
    try:
        bs = 64
        print(f"\n[Test 2] 256x256 Batch={bs}")
        x = torch.randn(bs, 3, 256, 256, device=device, dtype=dtype)
        with torch.no_grad():
            out = model(x)
        print(f"Output shape: {out.shape}")
        print_mem(f"After Batch {bs}")
        del x, out
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"[Test 2] Failed OOM: {e}")

    # Stress Test 3: 512x512 Batch 8
    try:
        bs = 8
        print(f"\n[Test 3] 512x512 Batch={bs}")
        x = torch.randn(bs, 3, 512, 512, device=device, dtype=dtype)
        with torch.no_grad():
            out = model(x)
        print(f"Output shape: {out.shape}")
        print_mem(f"After Batch {bs}")
        del x, out
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"[Test 3] Failed OOM: {e}")

    print("\nStress Test Complete!")

if __name__ == "__main__":
    demo()
