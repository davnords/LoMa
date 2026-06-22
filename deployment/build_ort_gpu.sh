#!/usr/bin/env bash
# Build onnxruntime-gpu from source for DGX Spark GB10 (sm_121, CUDA 13, aarch64).
# No prebuilt aarch64+CUDA13 wheel exists, so we compile one.
set -euo pipefail

REPO=/home/raman/Basir/LoMa
SRC="$REPO/.ortbuild/onnxruntime"
VENV_PY="$REPO/.venv/bin/python"
ORT_VER="v1.24.4"
CUDA_HOME="/usr/local/cuda"
CUDNN_HOME="/usr"

echo "[ort] $(date) starting build of onnxruntime-gpu $ORT_VER for sm_121"
mkdir -p "$REPO/.ortbuild"

if [ ! -d "$SRC/.git" ]; then
  echo "[ort] cloning $ORT_VER ..."
  git clone --recursive --branch "$ORT_VER" --depth 1 \
    https://github.com/microsoft/onnxruntime.git "$SRC"
else
  echo "[ort] source already present, reusing"
fi

cd "$SRC"
echo "[ort] building (this takes a while) ..."
"$VENV_PY" tools/ci_build/build.py \
  --build_dir build/Linux \
  --config Release \
  --parallel 12 \
  --nvcc_threads 1 \
  --use_cuda \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --build_shared_lib \
  --build_wheel \
  --skip_tests \
  --skip_submodule_sync \
  --compile_no_warning_as_error \
  --allow_running_as_root \
  --cmake_extra_defines \
      CMAKE_CUDA_ARCHITECTURES=121 \
      onnxruntime_BUILD_UNIT_TESTS=OFF

echo "[ort] build finished, locating wheel ..."
WHEEL=$(ls -t "$SRC"/build/Linux/Release/dist/onnxruntime_gpu-*.whl | head -1)
echo "[ort] wheel: $WHEEL"

echo "[ort] installing wheel into venv (replaces CPU onnxruntime) ..."
"$REPO/.venv/bin/pip" install --force-reinstall --no-deps "$WHEEL"

echo "[ort] verifying providers ..."
"$VENV_PY" - <<'PY'
import onnxruntime as ort
print("onnxruntime", ort.__version__)
print("providers:", ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers(), "CUDA EP missing!"
print("CUDAExecutionProvider OK")
PY

echo "[ort] $(date) DONE — onnxruntime-gpu built + installed + verified"
