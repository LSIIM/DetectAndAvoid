#!/bin/bash

set -e

# ============================================================
# OpenCV 4.10.0 + CUDA + cuDNN + GStreamer
# Jetson Orin NX / JetPack 6.2
# Python 3.10
# ============================================================

OPENCV_VERSION="4.10.0"
OPENCV_DIR="$HOME/opencv"
CONTRIB_DIR="$HOME/opencv_contrib"
BUILD_DIR="$OPENCV_DIR/build"

echo "============================================================"
echo " Instalacao do OpenCV $OPENCV_VERSION com CUDA"
echo " Jetson Orin NX - CUDA Architecture 8.7"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Dependencias
# ------------------------------------------------------------

echo "[1/7] Instalando dependencias..."

sudo apt update

sudo apt install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb-dev \
    libdc1394-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly

echo
echo "Dependencias instaladas."
echo

# ------------------------------------------------------------
# 2. Verificar Python
# ------------------------------------------------------------

echo "[2/7] Verificando Python..."

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_EXECUTABLE=$(which python3)

echo "Python encontrado: $PYTHON_EXECUTABLE"
echo "Versao: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo
    echo "ERRO: este script espera Python 3.10."
    echo "Python encontrado: $PYTHON_VERSION"
    exit 1
fi

PYTHON_INCLUDE_DIR="/usr/include/python3.10"
PYTHON_LIBRARY="/usr/lib/aarch64-linux-gnu/libpython3.10.so"
PYTHON_PACKAGES_PATH="$HOME/.local/lib/python3.10/site-packages"

echo

# ------------------------------------------------------------
# 3. Verificar CUDA
# ------------------------------------------------------------

echo "[3/7] Verificando CUDA..."

if ! command -v nvcc &> /dev/null; then
    echo
    echo "ERRO: nvcc nao encontrado."
    echo "Verifique se o CUDA do JetPack esta instalado."
    exit 1
fi

nvcc --version

echo

# ------------------------------------------------------------
# 4. Baixar OpenCV
# ------------------------------------------------------------

echo "[4/7] Baixando OpenCV $OPENCV_VERSION..."

if [ -d "$OPENCV_DIR" ]; then
    echo
    echo "A pasta $OPENCV_DIR ja existe."
    echo "Ela sera reutilizada."
else
    git clone \
        --branch "$OPENCV_VERSION" \
        --depth 1 \
        https://github.com/opencv/opencv.git \
        "$OPENCV_DIR"
fi

if [ -d "$CONTRIB_DIR" ]; then
    echo
    echo "A pasta $CONTRIB_DIR ja existe."
    echo "Ela sera reutilizada."
else
    git clone \
        --branch "$OPENCV_VERSION" \
        --depth 1 \
        https://github.com/opencv/opencv_contrib.git \
        "$CONTRIB_DIR"
fi

echo
echo "OpenCV e opencv_contrib preparados."
echo

# ------------------------------------------------------------
# 5. Configurar CMake
# ------------------------------------------------------------

echo "[5/7] Configurando CMake..."

cd "$OPENCV_DIR"

if [ -d "$BUILD_DIR" ]; then
    echo "Removendo build anterior..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH="$CONTRIB_DIR/modules" \
    \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=8.7 \
    -D CUDA_ARCH_PTX=8.7 \
    \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -D PYTHON3_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
    -D PYTHON3_LIBRARY="$PYTHON_LIBRARY" \
    -D PYTHON3_NUMPY_INCLUDE_DIRS="$PYTHON_PACKAGES_PATH/numpy/core/include" \
    -D PYTHON3_PACKAGES_PATH="$PYTHON_PACKAGES_PATH" \
    \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_opencv_java=OFF \
    -D BUILD_opencv_js=OFF

echo
echo "CMake configurado."
echo

# ------------------------------------------------------------
# 6. Compilar
# ------------------------------------------------------------

echo "[6/7] Compilando OpenCV..."
echo
echo "Usando 6 threads."
echo

make -j6

echo
echo "Compilacao concluida."
echo

# ------------------------------------------------------------
# 7. Instalar
# ------------------------------------------------------------

echo "[7/7] Instalando em /usr/local..."

sudo make install
sudo ldconfig

echo
echo "============================================================"
echo " Instalacao concluida!"
echo "============================================================"
echo

# ------------------------------------------------------------
# Informacoes finais
# ------------------------------------------------------------

echo "OpenCV instalado."
echo

echo "Para testar a instalacao diretamente:"
echo
echo "  PYTHONPATH=/usr/local/lib/python3.10/dist-packages:"
echo '  $PYTHONPATH python3 -c "import cv2; print(cv2.__version__)"'
echo

echo "Teste CUDA:"
echo
echo "  PYTHONPATH=/usr/local/lib/python3.10/dist-packages:"
echo '  $PYTHONPATH python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"'
echo

echo "Verificando CUDA/cuDNN/GStreamer..."
echo

PYTHONPATH="/usr/local/lib/python3.10/dist-packages:$PYTHONPATH" \
python3 -c "import cv2; print(cv2.getBuildInformation())" \
    | grep -E "NVIDIA CUDA|cuDNN|CUDA_ARCH|GStreamer|Python"

echo
echo "============================================================"
echo " Fim"
echo "============================================================"