# Phoenix Notes

On MI Phoenix nodes, set cache paths explicitly before Torch or JAX GPU runs:

```bash
export XDG_CACHE_HOME="/scratch/${USER}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
```

The same exports are captured in [`scripts/env/phoenix.sh`](../scripts/env/phoenix.sh).
