#!/usr/bin/env bash
# Project-local runtime setup. The cluster exports CUDA 12.3 libraries first;
# PyTorch 2.5.1+cu124 needs its bundled nvJitLink 12.4 library first.
A4_PROJECT_DIR="$(pwd -P)"
export PATH="$A4_PROJECT_DIR/.venv/bin:$PATH"
export LD_LIBRARY_PATH="$A4_PROJECT_DIR/.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
