#!/bin/bash

set -e

# ============================================================
# venv do DetectAndAvoid
# Python 3.10 / JetPack 6.1
# OpenCV CUDA deve ja estar instalado (utils/install_opencv_cuda.sh)
# ============================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/venv"
REQ="$ROOT/requirements.txt"

TORCH_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl"
TORCHVISION_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl"
ONNXRUNTIME_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl"

echo "============================================================"
echo " Setup da venv DetectAndAvoid"
echo " $VENV"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. venv
# ------------------------------------------------------------

echo "[1/7] Criando venv..."

if [ -d "$VENV" ]; then
    echo "A venv ja existe e sera reutilizada: $VENV"
else
    python3 -m venv "$VENV"
    echo "venv criada."
fi

PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
SITE="$VENV/lib/python3.10/site-packages"
mkdir -p "$SITE"

echo

# ------------------------------------------------------------
# 2. PyTorch e Torchvision (wheels JP6.1)
# ------------------------------------------------------------

echo "[2/7] Instalando PyTorch e Torchvision (sem dependencias)..."

"$PIP" install --no-deps "$TORCH_WHL"
"$PIP" install --no-deps "$TORCHVISION_WHL"

echo
echo "Wheels do torch instalados."
echo

# ------------------------------------------------------------
# 3. onnxruntime-gpu e NumPy
# ------------------------------------------------------------

echo "[3/7] Instalando onnxruntime-gpu e numpy==1.23.5..."

"$PIP" install --no-deps "$ONNXRUNTIME_WHL"
"$PIP" install --no-deps "numpy==1.23.5"

echo
echo "onnxruntime-gpu e numpy instalados."
echo

# ------------------------------------------------------------
# 4. Vincular OpenCV compilado com CUDA
# ------------------------------------------------------------

echo "[4/7] Vinculando OpenCV CUDA na venv..."

CV2_SRC=""

for candidate in \
    "$HOME/.local/lib/python3.10/site-packages/cv2" \
    "/usr/local/lib/python3.10/dist-packages/cv2" \
    "$HOME/.local/lib/python3.10/site-packages/cv2.so" \
    "/usr/local/lib/python3.10/dist-packages/cv2.so"
do
    if [ -e "$candidate" ]; then
        CV2_SRC="$candidate"
        break
    fi
done

if [ -z "$CV2_SRC" ]; then
    echo
    echo "ERRO: modulo cv2 com CUDA nao encontrado."
    echo "Rode antes: ./utils/install_opencv_cuda.sh"
    exit 1
fi

ln -sfn "$CV2_SRC" "$SITE/$(basename "$CV2_SRC")"

echo "cv2 ligado: $CV2_SRC -> $SITE/$(basename "$CV2_SRC")"
echo

# ------------------------------------------------------------
# 5. PyCUDA (compila contra o CUDA do JetPack)
# ------------------------------------------------------------

echo "[5/7] Compilando pycuda==2026.1..."

CUDA_INC=""
for candidate in \
    /usr/local/cuda/include/cuda.h \
    /usr/local/cuda/targets/aarch64-linux/include/cuda.h
do
    if [ -f "$candidate" ]; then
        CUDA_INC="$(dirname "$candidate")"
        break
    fi
done

if [ -z "$CUDA_INC" ]; then
    echo
    echo "ERRO: cuda.h nao encontrado."
    echo "Instale o CUDA toolkit do JetPack (nvcc e os headers)."
    exit 1
fi

export CUDA_INC_DIR="$CUDA_INC"
if [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_ROOT=/usr/local/cuda
    export PATH="/usr/local/cuda/bin:$PATH"
else
    export CUDA_ROOT="$(dirname "$CUDA_INC")"
fi
if [ -d /usr/local/cuda/lib64 ]; then
    export CUDA_LIB_DIR=/usr/local/cuda/lib64
fi

echo "CUDA headers: $CUDA_INC_DIR"
echo "nvcc: $(command -v nvcc || echo 'nao encontrado')"

"$PIP" install wheel
"$PIP" install --no-deps --no-build-isolation "pycuda==2026.1"

echo
echo "pycuda instalado."
echo

# ------------------------------------------------------------
# 6. requirements pinados, sem dependencias
# ------------------------------------------------------------

echo "[6/7] Instalando requirements.txt (sem dependencias)..."

"$PIP" install --no-deps -r "$REQ"

echo
echo "requirements instalados."
echo

# ------------------------------------------------------------
# 7. TensorRT do JetPack (nao o wheel CUDA 13 do PyPI)
# ------------------------------------------------------------

echo "[7/7] Vinculando TensorRT do JetPack na venv..."

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/venv"
SITE="$VENV/lib/python3.10/site-packages"
mkdir -p "$SITE"
echo "site-packages: $SITE"

TRT_SRC=""
for candidate in \
    /usr/lib/python3.10/dist-packages/tensorrt \
    /usr/local/lib/python3.10/dist-packages/tensorrt \
    /usr/lib/python3.10/dist-packages/tensorrt.so
do
    if [ -e "$candidate" ]; then
        TRT_SRC="$candidate"
        break
    fi
done

if [ -z "$TRT_SRC" ]; then
    echo
    echo "ERRO: modulo tensorrt do JetPack nao encontrado."
    echo "Procure em /usr/lib/python3.10/dist-packages/tensorrt"
    exit 1
fi

rm -rf "$SITE/tensorrt" "$SITE"/tensorrt-*.dist-info \
    "$SITE"/tensorrt_cu13_bindings* "$SITE"/tensorrt_cu13_libs* \
    "$SITE"/tensorrt_dispatch* "$SITE"/tensorrt_lean*
ln -sfn "$TRT_SRC" "$SITE/tensorrt"
if [ ! -e "$SITE/tensorrt" ]; then
    echo
    echo "ERRO: symlink do tensorrt nao foi criado em $SITE/tensorrt"
    exit 1
fi

echo "tensorrt ligado: $TRT_SRC -> $SITE/tensorrt"
echo
echo "============================================================"
echo " Setup concluido"
echo "============================================================"
echo
echo "Ative a venv:"
echo "  source venv/bin/activate"
echo
echo "Checagens:"
echo '  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
echo '  python -c "import onnxruntime; print(onnxruntime.__version__)"'
echo '  python -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"'
echo '  python -c "import tensorrt as trt; print(trt.__version__)"'
echo
echo "============================================================"
echo " Fim"
echo "============================================================"
