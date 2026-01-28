import torch
import os

ckpt_path = "/mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"

def inspect_ckpt():
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        return

    try:
        print(f"Loading {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
            
        keys = list(state_dict.keys())
        print(f"Total keys: {len(keys)}")
        
        # 1. FFN Structure
        swiglu_keys = [k for k in keys if "mlp.w1" in k]
        mlp_keys = [k for k in keys if "mlp.fc1" in k]
        print(f"\n[FFN Structure]")
        print(f"Has SwiGLU (w1/w2/w3)? {'YES' if swiglu_keys else 'NO'}")
        print(f"Has MLP (fc1/fc2)? {'YES' if mlp_keys else 'NO'}")
        
        # 2. Bias Check
        qkv_bias = any("qkv.bias" in k for k in keys)
        ffn_bias = any("mlp.w1.bias" in k or "mlp.fc1.bias" in k for k in keys)
        proj_bias = any("attn.proj.bias" in k for k in keys)
        print(f"\n[Bias Check]")
        print(f"Has QKV bias? {qkv_bias}")
        print(f"Has FFN bias? {ffn_bias}")
        print(f"Has Proj bias? {proj_bias}")
        
        # 3. LayerScale Check
        ls_keys = [k for k in keys if "ls1.gamma" in k or "ls2.gamma" in k]
        print(f"\n[LayerScale Check]")
        print(f"Has LayerScale (gamma)? {'YES' if ls_keys else 'NO'} ({len(ls_keys)} keys)")
        if ls_keys:
            print(f"Sample: {ls_keys[0]}")
            # Check value to guess init
            val = state_dict[ls_keys[0]]
            print(f"Sample value mean: {val.mean().item():.6f}")

        # 4. Register Tokens (DINOv2 feature)
        reg_tokens = any("register_tokens" in k for k in keys)
        print(f"\n[Other]")
        print(f"Has register_tokens? {reg_tokens}")

    except Exception as e:
        print(f"Failed to inspect: {e}")

if __name__ == "__main__":
    inspect_ckpt()
