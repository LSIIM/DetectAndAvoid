# DetectAndAvoid
Repositório com as vertentes estudadas no projeto de DetectAndAvoid

## Sistema Integrado

O sistema de integração principal combina detecção YOLO, Segmentação de Céu e processamento de Fluxo Óptico em um pipeline unificado.

### Uso

```bash
python main.py <video_path> [opções]
```

### Argumentos

- `--video_path` (opcional): Caminho para o arquivo de vídeo de entrada 
- `--clusters <num>` (opcional): Número de clusters para fluxo óptico (padrão: 5)
- `--confidence <conf>` (opcional): Limiar de confiança do YOLO (padrão: 0.6)
- `--output <caminho>` (opcional): Caminho do vídeo de saída
- `--resize-height <altura>` (opcional): Altura de redimensionamento do frame (padrão: 480)

### Exemplos

```bash
# Uso básico
python main.py videos/drone_video.mp4

# Com parâmetros personalizados
python main.py videos/drone_video.mp4 --clusters 3 --confidence 0.7

# Salvar vídeo de saída
python main.py videos/drone_video.mp4 --output processed_output.mp4
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
