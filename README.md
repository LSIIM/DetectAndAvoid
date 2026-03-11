# DeA2

Pipeline otimizado em Python para baixa latência com RTSP, YOLO, SkySeg e Optical Flow.

## Principais melhorias
- Captura em thread contínua com política `latest-frame` (descarta backlog).
- Workers assíncronos por módulo (YOLO / Sky / Flow).
- Intervalo independente por módulo (`--*-update-interval`).
- Gravação em tempo real (evita vídeo acelerado quando a IA processa menos FPS que a câmera).
- Opção headless (`--no-display`) para benchmark de throughput.

## Execução rápida
```bash
cd /home/gd60v1/Desktop/Algorithims/UFSC/DeA2
python -u main.py --video-ip 192.168.1.114
```

## Rodar com vídeo local
```bash
cd /home/gd60v1/Desktop/Algorithims/UFSC/DeA2
python -u main.py --video-file /caminho/para/video.mp4
```

## Preset focado em performance
```bash
python -u main.py \
  --video-ip 192.168.1.114 \
  --resize-height 320 \
  --yolo-update-interval 2 \
  --sky-update-interval 3 \
  --flow-update-interval 1 \
  --flow-gpu \
  --no-display
```

## Rodar em CPU (desativar CUDA)
```bash
python -u main.py \
  --video-file /caminho/para/video.mp4 \
  --disable-cuda \
  --yolo-model-path /caminho/para/modelo.pt
```

## Gravar saída sem acelerar vídeo
```bash
python -u main.py \
  --video-ip 192.168.1.114 \
  --output /home/gd60v1/Desktop/Algorithims/UFSC/DeA2/out.mp4 \
  --output-fps 30
```

## Flags úteis
- `--disable-sky`: desliga segmentação de céu.
- `--disable-flow`: desliga optical flow.
- `--disable-cuda`: força execução CPU-only no DeA2.
- `--flow-gpu`: usa `OpticalFlow/opticalflow_gpu.py`.
- `--stats-interval 2.0`: frequência de log de performance.
