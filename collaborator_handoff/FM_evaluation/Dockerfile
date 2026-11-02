FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /mnt/huawei_deepcad/dinov3
