import webdataset as wds
import numpy as np
import torch
import io

# 1. 自定义一个专门读取 .npy 的解码器
def npy_decoder(key, value):
    if not key.endswith(".npy"):
        return None
    return np.load(io.BytesIO(value))

# 2. 加载你生成的一个 .tar 文件
tar_path = "/mnt/huawei_deepcad/wds_shards/mixed_4ch-000178.tar"
dataset = wds.WebDataset(tar_path).decode(npy_decoder).to_tuple("npy", "__key__")

# 3. 遍历前几个样本进行硬性检查
print(f"Checking {tar_path}...")
for i, (tensor_array, key) in enumerate(dataset):
    if i >= 5: # 只检查前 5 个
        break
        
    print(f"\n--- Sample {i+1} ---")
    print(f"Filename Key: {key}")
    print(f"Shape: {tensor_array.shape}")
    print(f"Dtype: {tensor_array.dtype}")
    print(f"Max Val: {tensor_array.max()}, Min Val: {tensor_array.min()}")