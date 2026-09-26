FROM nvidia/cuda:12.8.2-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    HF_HOME=/opt/hf-cache

COPY code/requirements-eval.txt code/requirements-train.txt /tmp/
RUN pip install  --upgrade pip && \
    pip install  -r /tmp/requirements-eval.txt -r /tmp/requirements-train.txt

# Qwen's official multimodal SFT trainer, fixed to the same upstream commit as
# the vendored MMMU evaluation code. The local wrapper selects SDPA for 4090.
RUN git init /opt/qwen3-vl && \
    git -C /opt/qwen3-vl remote add origin https://github.com/QwenLM/Qwen3-VL.git && \
    git -C /opt/qwen3-vl sparse-checkout init --cone && \
    git -C /opt/qwen3-vl sparse-checkout set qwen-vl-finetune && \
    git -C /opt/qwen3-vl fetch --depth 1 origin 96588727e44c78b25ba03ea03b8e12f7e64fd0da && \
    git -C /opt/qwen3-vl checkout --detach FETCH_HEAD
ENV QWEN_TRAIN_ROOT=/opt/qwen3-vl/qwen-vl-finetune
COPY code/train/patch_upstream.py /tmp/patch_upstream.py
RUN python /tmp/patch_upstream.py

WORKDIR /opt/mmdl
COPY code/scripts/fetch_assets.sh code/scripts/fetch_hf_datasets.py code/scripts/
RUN bash code/scripts/fetch_assets.sh && python code/scripts/fetch_hf_datasets.py

COPY . .
RUN python -m compileall -q code && \
    python code/eval/tests/test_configuration.py && \
    python code/eval/tests/test_prompting.py && \
    python code/eval/tests/test_answer_extraction.py


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    git \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["bash"]