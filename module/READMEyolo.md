# Ultralytics YOLO no NVIDIA Jetson — **Executar no JetPack 6.1**

Este README descreve uma instalação **nativa** (sem Docker) para executar o Ultralytics YOLO no **JetPack 6.1** (Ubuntu 22.04 / Python 3.10 em `aarch64`).

> Dica: use um ambiente virtual para evitar conflitos de sistema:
>
> ```bash
> python3 -m venv venv && source ./venv/bin/activate
> ```

---

## 1) Atualizar o sistema e instalar o Ultralytics

1. Atualize e instale o `pip` mais recente:

```bash
sudo apt update
sudo apt install -y python3-pip
pip install -U pip
```

2. Instale o pacote Ultralytics com dependências de exportação:

```bash
pip install "ultralytics[export]"
```

3. Reinicie o dispositivo:

```bash
sudo reboot
```

---

## 2) Instalar **PyTorch** e **Torchvision** para JP6.1
Os binários padrão do PyPI para `torch`/`torchvision` **não** são compatíveis com Jetson (ARM64). Instale as rodas (wheels) pré‑compiladas para **JetPack 6.1** e **Python 3.10**:

```bash
# PyTorch 2.5.0 para JP6.1 (aarch64)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# Torchvision 0.20 para JP6.1 (aarch64)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

> **Compatibilidade**: consulte a página "PyTorch for Jetson" para outras combinações de JetPack/Python.

### 2.1) Corrigir dependência `cuSPARSELt` exigida pelo `torch 2.5`
Faça isso caso de algum problema relacionado a dependência
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

---

## 3) Instalar **onnxruntime-gpu** (aarch64)
O pacote do PyPI não fornece binários `aarch64` para Jetson. Use o wheel compatível com **JP6.1 + Python 3.10**:

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

> **Ajuste de NumPy**: após instalar `onnxruntime-gpu`, fixe o NumPy conforme abaixo para evitar incompatibilidades:

```bash
pip install numpy==1.23.5
```

---


## 5) Solução de problemas
- **Permissões do pip (Ubuntu 22.04)**: se instalar fora do `venv`, pode ser necessário `pip install --break-system-packages`.
- **Mismatch torch/torchvision**: valide se as versões acima correspondem ao seu JetPack e Python.
- **Para usar TensorRT**: no arquivo pyvenv.cfg, se desejar converter o modelo para TensorRT, adicione a seguinte linha:
```text
[sys]
include-system-site-packages = true 

```

---

# requirements-jetpack61.txt

Se preferir instalar via arquivo de requisitos, use o conteúdo abaixo e execute `pip install -r requirements-jetpack61.txt`:

```text
ultralytics[export]
# Wheels específicos para JP6.1 (Python 3.10, aarch64)
torch @ https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
torchvision @ https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
onnxruntime-gpu @ https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
numpy==1.23.5
```

---

**Referências**
- [Guia oficial Ultralytics — NVIDIA Jetson → *Run on JetPack 6.1*](https://docs.ultralytics.com/pt/guides/nvidia-jetson/#run-on-jetpack-61)


