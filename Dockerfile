# ===== 1. Baza: CUDA 11.0 + cuDNN8 (Ubuntu 20.04) =====
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

# ===== 2. Ustawienia =====
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# ===== 3. Systemowe zależności i Python 3.8 =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3-pip python3.8-distutils python3-venv \
    build-essential curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# Symlinki dla wygody
RUN ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

# ===== 4. Aktualizacja pip =====
RUN pip install --upgrade pip setuptools wheel

# ===== 5. PyTorch 1.7.1 + cu110 =====
# Uwaga: wersje muszą być sparowane: torchvision 0.8.2, torchaudio 0.7.2
RUN pip install \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ===== 6. PyG dopasowane do Torch 1.7.1 + cu110 =====
RUN pip install \
    torch-scatter==2.0.7 \
    torch-sparse==0.6.9 \
    torch-cluster==1.5.9 \
    torch-geometric==2.0.4 \
    -f https://data.pyg.org/whl/torch-1.7.1+cu110.html

# ===== 7. Reszta bibliotek =====
RUN pip install class-resolver==0.3.10 ndlib==5.1.1 geopy==2.1.0 \
    matplotlib scikit-learn seaborn

# ===== 8. Kopiowanie kodu (na końcu dla lepszego cache) =====
COPY . /app/

# ===== 9. Domyślne wejście =====
CMD ["/bin/bash"]