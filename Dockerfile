FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
LABEL maintainer="nesy-course"

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workdir
ENV APP_PATH=/workdir
ENV TORCH_CUDA_ARCH_LIST="12.0"

RUN apt-get update -y && \
    apt-get install -y curl \
                       git \
                       bash \
                       nano \
                       wget \
                       unzip \
                       build-essential \
                       python3.12 \
                       python3-pip \
                       python3.12-venv && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch with CUDA 12.8 for RTX 5090 (sm_120) -- must be 2.7.0+, earlier
# builds don't have Blackwell kernels compiled in regardless of driver.
RUN pip install --no-cache-dir \
    torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=dialog