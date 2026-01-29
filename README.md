# DetectAndAvoid
Repositório com as vertentes estudadas no projeto de DetectAndAvoid

## Sistema Integrado

O sistema de integração principal combina detecção YOLO, Segmentação de Céu e processamento de Fluxo Óptico em um pipeline unificado.

## 1) Instalar **PyTorch** e **Torchvision** para JP6.1
Os binários padrão do PyPI para `torch`/`torchvision` **não** são compatíveis com Jetson (ARM64). Instale as rodas (wheels) pré‑compiladas para **JetPack 6.1** e **Python 3.10**:

```bash
# PyTorch 2.5.0 para JP6.1 (aarch64)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# Torchvision 0.20 para JP6.1 (aarch64)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

> **Compatibilidade**: consulte a página "PyTorch for Jetson" para outras combinações de JetPack/Python.

### 1.1) Corrigir dependência `cuSPARSELt` exigida pelo `torch 2.5`
Faça isso caso de algum problema relacionado a dependência
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

---

## 2) Instalar **onnxruntime-gpu** (aarch64)
O pacote do PyPI não fornece binários `aarch64` para Jetson. Use o wheel compatível com **JP6.1 + Python 3.10**:

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

> **Ajuste de NumPy**: após instalar `onnxruntime-gpu`, fixe o NumPy conforme abaixo para evitar incompatibilidades:

```bash
pip install numpy==1.23.5
```

---


## 3) Instalar dependências restantes

```bash
pip install -r requirements.txt
```


### 4) Gerar .engine
Arquivo otimizado do modelo YOLO para GPU.
```bash
python .\Yolo\Yolo11\yolo_to_TensorRT.py
```

### Uso

```bash
python main.py <video_path> [opções]
```

### Argumentos

- `--video-ip <ip>` (opcional): Endereço de IP da câmera (padrão:192.168.144.25) 
- `--clusters <num>` (opcional): Número de clusters para fluxo óptico (padrão: 5)
- `--confidence <conf>` (opcional): Limiar de confiança do YOLO (padrão: 0.6)
- `--output <caminho>` (opcional): Caminho do vídeo de saída
- `--resize-height <altura>` (opcional): Altura de redimensionamento do frame (padrão: 480)

### Exemplos

```bash
# Uso básico
python main.py

# Com parâmetros personalizados
python main.py --video-ip 192.168.144.25 --clusters 3 --confidence 0.7

# Salvar vídeo de saída
python main.py --video-ip 192.168.144.25 --output processed_output.mp4
```

### Controles

- `ESC` ou `q`: Sair do processamento
- `s`: Salvar frame atual como imagem

### Visualização

O sistema mostra três visualizações processadas lado a lado:
1. **Detecção YOLO**: Detecção de objetos com caixas delimitadoras
2. **Segmentação de Céu**: Detecção de horizonte e segmentação do céu
3. **Fluxo Óptico**: Agrupamento de movimento e vetores de velocidade

## Módulos Individuais
