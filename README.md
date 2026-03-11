# DeACpp

Conversao do pipeline `DeA2/main.py` para C++.

## O que foi portado
- Captura assíncrona com politica `latest-frame`.
- Workers assíncronos por modulo (YOLO / SkySeg / OpticalFlow).
- Loop principal com estatisticas, visualizacao combinada e gravacao em tempo real.
- CLI equivalente ao `DeA2` (com os mesmos argumentos principais).

## Suporte de modelos (autocontido em `DeACpp/models`)
- YOLO:
  - default: `best_yolo_11_JUNHO_nano_drones_DGX_rebuilt.engine`.
  - fallback: `.onnx` via OpenCV DNN, com tentativa de rebuild TensorRT quando parser estiver disponivel.
- SkySeg:
  - default: `skyseg_fp16_trt_sm87.engine` (TensorRT).
  - fallback: `skyseg_fp16.onnx` (TensorRT build/cache local em `DeACpp/trt_cache` ou OpenCV DNN).
- OpticalFlow:
  - default: LK em CPU.
  - com `--flow-gpu`: usa VPI OpticalFlowPyrLK (CUDA) quando `libnvvpi` estiver disponivel; fallback automatico para CPU em caso de falha.

## Build
```bash
cd /home/gd60v1/Desktop/Algorithims/UFSC/DeACpp
cmake -S . -B build
cmake --build build -j
```

## Execucao rapida
```bash
cd /home/gd60v1/Desktop/Algorithims/UFSC/DeACpp
./build/deacpp --video-file /caminho/video.mp4
```

## Execucao com benchmark puro (sem renderizacao)
```bash
./build/deacpp \
  --video-file /caminho/video.mp4 \
  --no-display
```

## RTSP otimizado (Jetson)
```bash
./build/deacpp \
  --video-ip 192.168.1.114 --video-port 1945 --video-path / \
  --rtsp-backend gstreamer \
  --rtsp-transport tcp \
  --rtsp-latency-ms 60 \
  --rtsp-open-timeout-ms 2500 \
  --rtsp-first-frame-timeout 20 \
  --rtsp-max-timeouts 8
```
- `--rtsp-backend auto` tenta GStreamer (NVDEC) e faz fallback para FFmpeg.
- `--rtsp-backend ffmpeg` força caminho legado.
- `--rtsp-open-timeout-ms` reduz tempo preso em tentativa de conexao ruim.
- `--rtsp-first-frame-timeout` evita abortar cedo enquanto o stream ainda aquece.
- `--rtsp-max-timeouts` controla quantos timeouts consecutivos de leitura sao tolerados.

### Troubleshooting RTSP/GStreamer
- Erro `Internal data stream error` em `rtspsrc` geralmente ocorre quando o stream possui audio+video e o pipeline nao filtra o pad de video.
- O `DeACpp` agora filtra `application/x-rtp,media=video` no caminho GStreamer para evitar esse problema.
- Teste rapido de descoberta do stream:
```bash
gst-discoverer-1.0 rtsp://192.168.1.114:1945/
```

## Observacao importante sobre .engine
Arquivos `.engine` sao especificos de GPU/driver/CUDA/TensorRT. Se o engine tiver sido gerado em ambiente diferente, a desserializacao pode falhar. Nesse caso, use um `.onnx` ou reexporte o `.engine` no mesmo ambiente-alvo.
