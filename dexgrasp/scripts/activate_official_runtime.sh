#!/usr/bin/env bash
# Source this file to use the small, isolated AnyDexGrasp official runtime.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source scripts/activate_official_runtime.sh" >&2
  exit 2
fi

_DEXGRASP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ANYDEX_VENV="$_DEXGRASP_ROOT/.venv-anydex-official"
_TORCH_PACKAGE="/home/qiaoguanren/anaconda3/pkgs/pytorch-1.13.1-py3.8_cuda11.6_cudnn8.3.2_0"
_TORCH_SITE="$_TORCH_PACKAGE/lib/python3.8/site-packages"
_TORCH_LIB="$_TORCH_SITE/torch/lib"

if [[ ! -x "$_ANYDEX_VENV/bin/python" ]]; then
  echo "missing official runtime: $_ANYDEX_VENV" >&2
  return 1
fi
if [[ ! -d "$_TORCH_SITE/torch" ]]; then
  echo "missing cached PyTorch 1.13.1 package: $_TORCH_SITE" >&2
  return 1
fi

export DEXGRASP_OFFICIAL_PYTHON="$_ANYDEX_VENV/bin/python"
export PATH="$_ANYDEX_VENV/bin:$PATH"
export PYTHONPATH="$_TORCH_SITE:$_DEXGRASP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$_TORCH_LIB:/home/qiaoguanren/anaconda3/pkgs/cuda-cudart-11.6.55-he381448_0/lib:/home/qiaoguanren/anaconda3/pkgs/cuda-nvtx-11.6.124-h0630a44_0/lib:/home/qiaoguanren/anaconda3/pkgs/libcublas-11.9.2.110-h5e84587_0/lib:/home/qiaoguanren/cuda-11.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/home/qiaoguanren/cuda-11.6"
export TORCH_CUDA_ARCH_LIST="8.6"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

unset _DEXGRASP_ROOT _ANYDEX_VENV _TORCH_PACKAGE _TORCH_SITE _TORCH_LIB
