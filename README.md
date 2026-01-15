# DetectAndAvoid
Repositório com as vertentes estudadas no projeto de DetectAndAvoid

## Sistema Integrado

O sistema de integração principal combina detecção YOLO, Segmentação de Céu e processamento de Fluxo Óptico em um pipeline unificado.

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
