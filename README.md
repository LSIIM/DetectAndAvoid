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

### Saída JSON para integração

Quando a opção `--verbose` está habilitada, o sistema imprime um JSON por frame. O
mesmo registro também é gravado no arquivo de log configurado quando `--log-path` está habilitado:

```json
{
  "frame": 2301,
  "timestamp": 1792586184.0115013,
  "tracks": [
    {
      "track_id": 53,
      "classe": "drone",
      "confianca": 0.8344,
      "cx": 0.6567,
      "cy": 0.2835,
      "vx": -5.1081,
      "vy": -63.2217,
      "S": 0.0,
      "deltaS": 0.000965,
      "timestamp": 1792586184.0115013
    }
  ],
  "aproximacoes": [
    {
      "track_id": 53,
      "indice": 1
    }
  ]
}
```

Formato esperado dos campos:

| Campo | Tipo/formato | Descrição |
| --- | --- | --- |
| `frame` | inteiro (`int`) | Número sequencial do frame processado. |
| `timestamp` | número real (`float`) | Instante do frame em segundos desde o Unix epoch (UTC). |
| `tracks` | array de objetos | Lista das detecções/rastreamentos ativos no frame. Pode ser `[]`. |
| `track_id` | inteiro (`int`) | Identificador estável do objeto enquanto ele permanecer rastreado. |
| `classe` | texto (`string`) ou `null` | Classe detectada, por exemplo `"drone"`. Pode ser `null` antes da classificação ser associada ao rastreamento. |
| `confianca` | número real (`float`) ou `null` | Confiança da detecção, normalmente entre `0.0` e `1.0`. |
| `cx`, `cy` | número real (`float`) | Centro do objeto em pixels no frame redimensionado que é processado pelo sistema; `cx` cresce da esquerda para a direita e `cy` de cima para baixo. |
| `vx`, `vy` | número real (`float`) | Velocidade estimada do centro em pixels por segundo. Valores negativos indicam movimento para a esquerda ou para cima, respectivamente. |
| `S` | número real (`float`) ou `null` | Valor de profundidade bruto do ZipDepth na região do objeto, sem normalização adicional. A unidade depende do modelo de profundidade. |
| `deltaS` | número real (`float`) ou `null` | Variação de `S` entre as janelas histórica mais antiga e mais recente. Fica `null` enquanto não houver histórico suficiente. |
| `aproximacoes` | array de objetos | Lista de aproximações identificadas no frame. Pode ser `[]`. |
| `indice` | inteiro (`int`) | Índice ou classificação da aproximação definida pelo módulo responsável. |

Os dois campos `timestamp` (no nível raiz e dentro de cada item de `tracks`)
representam o mesmo instante do frame. Sistemas consumidores devem aceitar tanto
`tracks: []` quanto `aproximacoes: []`, pois um frame pode não conter objetos
rastreados ou aproximações.

## Módulos Individuais
