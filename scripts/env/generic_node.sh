#!/usr/bin/env bash
set -euo pipefail

export XDG_CACHE_HOME="/scratch/${USER}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"

echo "Cache environment configured."
