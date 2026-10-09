# DetectAndAvoid
Repositório com as vertentes estudadas no projeto de DetectAndAvoid

## Sistema Integrado

O sistema de integração principal combina detecção YOLO, ZipDepth e Fluxo Óptico em um pipeline unificado.

JetPack 6.1 e Python 3.10. Rode os passos nesta ordem.

## 1) Clonar o repositório

```bash
git clone https://github.com/LSIIM/DetectAndAvoid.git
cd DetectAndAvoid
```

## 2) Instalar OpenCV compilado com CUDA

O script compila o OpenCV 4.10 com CUDA, cuDNN e GStreamer. Leva vários minutos e deve rodar antes da venv.

```bash
chmod +x utils/install_opencv_cuda.sh
./utils/install_opencv_cuda.sh
```

## 3) Setup da venv

Cria `venv/` na raiz do projeto. Instala os wheels de PyTorch, Torchvision e onnxruntime-gpu para Jetson (sem puxar dependências do PyPI), vincula o OpenCV CUDA e o TensorRT do JetPack, e instala o `requirements.txt` com `--no-deps`.

```bash
chmod +x utils/setup_venv.sh
./utils/setup_venv.sh
source venv/bin/activate
```

> **cuSPARSELt**: se o `torch 2.5` falhar por essa dependência:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

Checagens, com a venv ativa:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"
python -c "import tensorrt as trt; print(trt.__version__)"
```

## 4) Gerar engines TensorRT

Exporta o YOLO (`.engine`) e o ZipDepth (`.trt`) para `weights/`. O YOLO termina e libera a memória da Jetson antes do ZipDepth. Precisa dos arquivos `weights/best_yolo26_drone_bird_aircraft_junho_2026.pt` e `weights/zipdepth_base_384x384.onnx`.

```bash
python ./utils/export_weights_trt.py
```

## 5) Uso

Com a venv ativa:

```bash
python main.py [flags]
```

### Argumentos

- `--video-ip <ip>` (opcional): Endereço de IP da câmera (padrão: 192.168.144.25)
- `--video-path <path>` (opcional): Caminho do vídeo a ser processado, no lugar da câmera
- `--clusters <num>` (opcional): Número de clusters para fluxo óptico (padrão: 5)
- `--confidence <conf>` (opcional): Limiar de confiança do YOLO (padrão: 0.6)
- `--output <caminho>` (opcional): Caminho do vídeo de saída
- `--resize-height <altura>` (opcional): Altura de redimensionamento do frame (padrão: 480)
- `--yolo-model-path <caminho>` (opcional): Engine YOLO (padrão: `weights/best_yolo26_drone_bird_aircraft_junho_2026.engine`)
- `--depth-model-path <caminho>` (opcional): Engine ZipDepth (padrão: `weights/zipdepth_base_384x384_fp16.trt`)
- `--no-display` (opcional): Não abre `cv2.imshow`. O `--output` continua valendo
- `--visual-depth` (opcional): Coloriza o mapa de profundidade e grava o vídeo lado a lado (YOLO + fluxo | ZipDepth)
- `--verbose` (opcional): Imprime o JSON por frame e os tempos de cada etapa

### Exemplos

```bash
# Câmera no IP padrão
python main.py

# Recomendado para uso em testes de voo
python main.py --no-display

# Vídeo gravado, sem janela, com log por frame
python main.py --video-path Videos/droneVSdrone1.mp4 --output c.mp4 --no-display --verbose

# Vídeo de debug com profundidade colorida ao lado
python main.py --video-path REC001.mp4 --output debug.mp4 --visual-depth

# Câmera, confiança e clusters
python main.py --video-ip 192.168.144.25 --clusters 3 --confidence 0.7 --output processed_output.mp4
```

### Controles

- `ESC` ou `q`: Sair do processamento
- `s`: Salvar frame atual como imagem

### Visualização

Sem `--visual-depth`, a janela mostra YOLO e fluxo óptico no mesmo frame. Com `--visual-depth`, o vídeo fica lado a lado: à esquerda YOLO e fluxo óptico, à direita o ZipDepth colorido.

## Módulos Individuais
