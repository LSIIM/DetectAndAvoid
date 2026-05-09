import torch
import os
import time
import cv2 as cv
import numpy as np
import onnxruntime
from ultralytics import YOLO
from collections import deque
from modules.yolo_module import YOLODetector
from modules.sky_seg_module import SkySegmentation

# ============================= CONFIGURAÇÕES =============================
# Caminhos
VIDEO_PATH = r"Raw_Videos/droneVSdrone1.mp4"

#YOLO_MODEL_PATH = r"Weights/best_yolo_11_JUNHO_nano_drones_DGX.pt"
YOLO_MODEL_PATH = r"Weights/best_yolo_11_JUNHO_nano_drones_DGX.engine"

HORIZON_MODEL_PATH = "Weights/skyseg_fp16.onnx"
USE_TENSORRT_SKYSEG = True

TRACKER_CONFIG = "bytetrack.yaml"

# Configurações de processamento
YOLO_CONFIDENCE = 0.5
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 640
HORIZON_MODEL_INPUT_SIZE = (320, 320)
SEGMENTATION_UPDATE_INTERVAL = 30  # Atualiza segmentação a cada N frames

# Configurações do sistema de alerta de aproximação
TRAIL_LENGTH = 50
APPROACH_AREA_INCREASE_THRESHOLD = 1.1  # 10% de aumento
ALERT_DURATION = 1.5  # segundos
ALERT_MESSAGE = "# ALERTA: APROXIMACAO DETECTADA"
ALERT_TEXT_COLOR = (0, 0, 255)  # vermelho
ALERT_BOX_COLOR = (0, 0, 0)  # fundo preto
ALERT_FONT_SCALE = 1
ALERT_THICKNESS = 2

# Configurações de análise de direção do voo
SAMPLE_AREA_SIZE = 30  # Tamanho da área de amostragem no centro
SKY_UPPER_THRESHOLD = 0.75  # Limiar para detectar SUBINDO
SKY_LOWER_THRESHOLD = 0.25  # Limiar para detectar DESCENDO
BINARY_THRESHOLD = 128  # Limiar para binarização da máscara

# Configuração de saída
input_dir, filename = os.path.split(VIDEO_PATH)
name, ext = os.path.splitext(filename)
OUTPUT_PATH = os.path.join(input_dir, f"{name}_final_processado{ext}")



# ============================= FUNÇÃO PRINCIPAL =============================
def main():
    print("=" * 60)
    print("SISTEMA UNIFICADO DE DETECÇÃO E ANÁLISE DE VOO")
    print("=" * 60)
    print("\n--- Configurando Modelos ---")
    
    # Inicializar detectores
    try:
        yolo_detector = YOLODetector(
            model_path=YOLO_MODEL_PATH,
            tracker_config=TRACKER_CONFIG,
            confidence_threshold=YOLO_CONFIDENCE,
            trail_length=TRAIL_LENGTH,
            approach_threshold=APPROACH_AREA_INCREASE_THRESHOLD,
            alert_duration=ALERT_DURATION,
            alert_message=ALERT_MESSAGE,
            alert_text_color=ALERT_TEXT_COLOR,
            alert_box_color=ALERT_BOX_COLOR,
            alert_font_scale=ALERT_FONT_SCALE,
            alert_thickness=ALERT_THICKNESS
        )
        
        sky_segmentation = SkySegmentation(
            model_path=HORIZON_MODEL_PATH,
            input_size=HORIZON_MODEL_INPUT_SIZE,
            update_interval=SEGMENTATION_UPDATE_INTERVAL,
            sample_area_size=SAMPLE_AREA_SIZE,
            sky_upper_threshold=SKY_UPPER_THRESHOLD,
            sky_lower_threshold=SKY_LOWER_THRESHOLD,
            binary_threshold=BINARY_THRESHOLD,
            use_tensorrt=USE_TENSORRT_SKYSEG 
        )
    except Exception as e:
        print(f"✗ ERRO na inicialização: {e}")
        return
    
    # Abrir vídeo
    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"✗ ERRO ao abrir vídeo: {VIDEO_PATH}")
        return
    
    # Configurar vídeo de saída
    video_fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_video = cv.VideoWriter(OUTPUT_PATH, fourcc, video_fps,
                              (PROCESSING_WIDTH * 2, PROCESSING_HEIGHT))
    
    # Variáveis de estatísticas
    frame_count = 0
    total_processing_start_time = time.time()
    
    print("\n--- Iniciando Processamento ---")
    print(f"Vídeo de entrada : {os.path.basename(VIDEO_PATH)}")
    print(f"Vídeo de saída   : {os.path.basename(OUTPUT_PATH)}")
    print(f"Total de frames  : {total_frames}")
    print(f"FPS do vídeo     : {video_fps:.2f}")
    print("\nProcessando...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        frame_count += 1
        
        # Redimensionar frame
        resized_frame = cv.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT),
                                 interpolation=cv.INTER_AREA)
        
        # ===== PROCESSAR COM YOLO =====
        yolo_frame, approach_detected = yolo_detector.process_frame(resized_frame)
        
        # ===== PROCESSAR SEGMENTAÇÃO DO CÉU =====
        sky_frame, flight_status, sky_ratio = sky_segmentation.process_frame(resized_frame)
        
        # ===== ADICIONAR FPS NO FRAME YOLO =====
        frame_end_time = time.time()
        instant_fps = 1.0 / (frame_end_time - frame_start_time) if frame_end_time > frame_start_time else 0
        cv.putText(yolo_frame, f"FPS: {instant_fps:.1f}", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ===== COMBINAR E SALVAR FRAMES =====
        combined_frame = np.hstack((yolo_frame, sky_frame))
        out_video.write(combined_frame)
        
        # Mostrar preview (opcional)
        cv.imshow("YOLO Detection | Sky Segmentation & Flight Analysis", combined_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("\nProcessamento interrompido pelo usuário.")
            break
        
        # Atualizar progresso
        if frame_count % 30 == 0:
            elapsed_time = time.time() - total_processing_start_time
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            eta = ((elapsed_time / frame_count) * (total_frames - frame_count)) if frame_count > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(f"Progresso: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"FPS médio: {avg_fps:.2f} | ETA: {eta:.1f}s")
    
    # ===== FINALIZAÇÃO =====
    total_processing_end_time = time.time()
    total_time = total_processing_end_time - total_processing_start_time
    average_fps = frame_count / total_time if total_time > 0 else 0
    
    cap.release()
    out_video.release()
    cv.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)
    print(f"✓ Frames processados         : {frame_count}")
    print(f"✓ Tempo total               : {total_time:.2f} segundos")
    print(f"✓ FPS médio de processamento: {average_fps:.2f}")
    print(f"✓ Vídeo salvo em            : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()