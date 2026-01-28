# DINOv3 Distributed Feature Extraction (Stage 2)

This project implements a distributed pipeline for extracting image features using DINOv3 on a Ray cluster.

## Requirements

- Python 3.8+
- PyTorch (with CUDA support)
- Ray
- OpenCV (`opencv-python`)
- Rich
- DINOv3 (located at `/mnt/huawei_deepcad/dinov3`)

## Structure

- `main.py`: Driver script.
- `actors.py`: Ray Actor implementation.
- `processor.py`: Image processing logic.

## Setup & Running

**Important**: Do NOT run this from the `dinov3` package root directly to avoid conflicts with the local `logging` module. 
We have placed these scripts in the `feature_extraction` directory.

### 1. Running the Whole Dataset with 'Crop' Strategy

Navigate to the `feature_extraction` folder or run from the repo root:

```bash
# Run from inside feature_extraction
cd feature_extraction
python main.py \
  --input_dir /mnt/huawei_deepcad/onepb/shards_task2 \
  --output_dir /mnt/huawei_deepcad/largedata_npy/ \
  --ar_strategy crop
```

### 2. Running a Specific Subfolder

```bash
cd feature_extraction
python main.py \
  --input_dir /mnt/huawei_deepcad/onepb/shards_task2/standard/ \
  --output_dir /mnt/huawei_deepcad/largedata_npy/ \
  --ar_strategy tile \
  --checkpoint_path /mnt/deepcad_nfs/xuzijing/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
```

## Implementation Details

- **Resolution**: 256x256 (Small/Medium) or 512x512 (Large).
- **Huge Images**: Tiled into 512x512 patches, inferred, and mean-pooled.
- **Ray**: Distributed across all available GPUs.
