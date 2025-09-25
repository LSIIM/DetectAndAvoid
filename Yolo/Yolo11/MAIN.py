import torch
import os
import time
import cv2 as cv
import numpy as np
import onnxruntime
from ultralytics import YOLO
from collections import deque

# ============================= CONFIGURAÇÕES =============================
# Caminhos
VIDEO_PATH = r"videos_test\droneVSdrone1.mp4"
YOLO_MODEL_PATH = r"Weights\yolo_11_JUNHO_nano_drones_DGX.pt"
HORIZON_MODEL_PATH = "Weights\skyseg_fp16.onnx"
TRACKER_CONFIG = "bytetrack.yaml"

# Configurações de processamento
YOLO_CONFIDENCE = 0.5
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 480
HORIZON_MODEL_INPUT_SIZE = (320, 320)
SEGMENTATION_UPDATE_INTERVAL = 20  # Atualiza segmentação a cada N frames

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
OUTPUT_PATH = os.path.join(input_dir, f"{name}_unified_complete{ext}")

# ============================= FUNÇÕES AUXILIARES =============================

def calculate_area(box):
    """Calcula a área de uma caixa delimitadora."""
    x1, y1, x2, y2 = box
    return abs((x2 - x1) * (y2 - y1))

def run_onnx_inference(onnx_session, input_size_hw, image_bgr):
    """Executa inferência do modelo ONNX para segmentação do céu."""
    original_height, original_width = image_bgr.shape[:2]

    # Redimensionar e preparar imagem
    resized_image = cv.resize(
        image_bgr,
        dsize=(input_size_hw[1], input_size_hw[0]),
        interpolation=cv.INTER_AREA
    )

    # Converter para RGB e normalizar
    rgb_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
    normalized_image = np.array(rgb_image, dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_image = (normalized_image / 255.0 - mean) / std
    
    # Transpor e preparar tensor
    transposed_image = normalized_image.transpose(2, 0, 1)
    input_tensor = transposed_image.reshape(1, 3, input_size_hw[0], input_size_hw[1]).astype(np.float16)

    # Executar inferência
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: input_tensor})
    output_mask = np.array(onnx_result).squeeze()

    # Normalizar saída
    min_val, max_val = np.min(output_mask), np.max(output_mask)
    if max_val > min_val:
        output_mask = (output_mask - min_val) / (max_val - min_val)
    else:
        output_mask = np.zeros_like(output_mask)

    output_mask_uint8 = (output_mask * 255).astype('uint8')

    # Redimensionar para tamanho original
    return cv.resize(
        output_mask_uint8,
        (original_width, original_height),
        interpolation=cv.INTER_NEAREST
    )

def analyze_flight_direction(binary_mask):
    """
    Analisa a direção do voo baseado na proporção de céu no centro da imagem.
    Retorna: (status, sky_ratio, roi_coords)
    """
    height, width = binary_mask.shape[:2]
    center_y, center_x = height // 2, width // 2
    half_size = SAMPLE_AREA_SIZE // 2
    
    # Definir região de interesse (ROI)
    y_start = max(0, center_y - half_size)
    y_end = min(height, center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(width, center_x + half_size)
    
    # Calcular proporção de céu na ROI
    center_roi = binary_mask[y_start:y_end, x_start:x_end]
    sky_ratio = np.mean(center_roi) / 255.0
    
    # Determinar status do voo
    if sky_ratio > SKY_UPPER_THRESHOLD:
        status = "SUBINDO"
        color = (0, 255, 255)  # amarelo
    elif sky_ratio < SKY_LOWER_THRESHOLD:
        status = "DESCENDO"
        color = (255, 0, 0)    # azul
    else:
        status = "NIVELADO"
        color = (0, 255, 0)    # verde
    
    return status, sky_ratio, (x_start, y_start, x_end, y_end), color

def draw_center_cross(image, size=20, color=(0, 0, 255), thickness=2):
    """Desenha uma cruz no centro da imagem."""
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    cv.line(image, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
    cv.line(image, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

def draw_flight_status(frame, status, sky_ratio, roi_coords, color):
    """Desenha informações de status do voo na imagem."""
    x_start, y_start, x_end, y_end = roi_coords
    
    # Desenhar ROI
    cv.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 3)
    
    # Desenhar cruz central
    draw_center_cross(frame, size=15, color=color, thickness=2)
    
    # Adicionar texto com status
    status_text = f"VOO: {status}"
    ratio_text = f"CEU: {sky_ratio:.1%}"
    
    # Posicionar textos no canto superior direito
    text_x = frame.shape[1] - 200
    cv.putText(frame, status_text, (text_x, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv.putText(frame, ratio_text, (text_x, 55), 
               cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_yolo_detections(frame, yolo_model, tracker_config, confidence_threshold,
                          track_history, track_colors, global_max_area):
    """
    Processa detecções YOLO com tracking.
    Retorna: (frame_processado, approach_detected, novo_global_max_area)
    """
    # Executar YOLO com tracking
    results = yolo_model.track(frame, persist=True, tracker=tracker_config, 
                               verbose=False, conf=confidence_threshold)
    
    approach_detected = False
    current_frame_max_area = 0.0
    
    if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        
        if results[0].boxes.conf is not None:
            confidences = results[0].boxes.conf.cpu().numpy()
        else:
            confidences = np.zeros(len(boxes), dtype=float)
        
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
        else:
            ids = list(range(len(boxes)))
        
        # Calcular área máxima do frame atual
        for box in boxes:
            area = calculate_area(box)
            if area > current_frame_max_area:
                current_frame_max_area = area
        
        # Verificar aproximação
        if global_max_area > 0 and current_frame_max_area > global_max_area * APPROACH_AREA_INCREASE_THRESHOLD:
            approach_detected = True
        
        # Atualizar recorde global
        new_global_max_area = max(global_max_area, current_frame_max_area)
        
        # Desenhar detecções e trilhas
        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            tid = ids[idx] if idx < len(ids) else -1
            x1, y1, x2, y2 = box
            
            # Desenhar caixa
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adicionar label
            label_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
            cv.putText(frame, f"ID:{tid} {conf:.2f}", label_pos,
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Atualizar histórico de tracking
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if tid not in track_history:
                track_history[tid] = deque(maxlen=TRAIL_LENGTH)
                track_colors[tid] = (
                    int(np.random.randint(50, 255)),
                    int(np.random.randint(50, 255)),
                    int(np.random.randint(50, 255))
                )
            track_history[tid].append((cx, cy))
            
            # Desenhar trilha
            pts = np.array(track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) > 1:
                cv.polylines(frame, [pts], False, track_colors[tid], 2)
    else:
        new_global_max_area = global_max_area
    
    return frame, approach_detected, new_global_max_area

def draw_alert(frame, last_approach_time):
    """Desenha alerta de aproximação se necessário."""
    if time.time() < last_approach_time + ALERT_DURATION:
        (tw, th), baseline = cv.getTextSize(ALERT_MESSAGE,
                                           cv.FONT_HERSHEY_SIMPLEX,
                                           ALERT_FONT_SCALE,
                                           ALERT_THICKNESS)
        pad = 5
        x1a, y1a = 15 - pad, 80 - th - pad
        x2a, y2a = 15 + tw + pad, 80 + baseline + pad
        
        cv.rectangle(frame, (x1a, y1a), (x2a, y2a), ALERT_BOX_COLOR, -1)
        cv.putText(frame, ALERT_MESSAGE, (15, 80),
                  cv.FONT_HERSHEY_SIMPLEX,
                  ALERT_FONT_SCALE,
                  ALERT_TEXT_COLOR,
                  ALERT_THICKNESS,
                  cv.LINE_AA)

# ============================= FUNÇÃO PRINCIPAL =============================

def main():
    print("=" * 60)
    print("SISTEMA UNIFICADO DE DETECÇÃO E ANÁLISE DE VOO")
    print("=" * 60)
    print("\n--- Configurando Modelos ---")
    
    # Verificar GPU para YOLO
    if not torch.cuda.is_available():
        print("ERRO: PyTorch não encontrou uma GPU compatível com CUDA.")
        print("O modelo YOLO não pode ser executado na GPU. Encerrando.")
        return
    
    # Carregar modelo YOLO
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH).to("cuda")
        print(f"✓ Modelo YOLO carregado na GPU")
    except Exception as e:
        print(f"✗ ERRO ao carregar modelo YOLO: {e}")
        return
    
    # Carregar modelo de segmentação
    try:
        onnx_session = onnxruntime.InferenceSession(
            HORIZON_MODEL_PATH, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print(f"✓ Modelo de segmentação carregado: {onnx_session.get_providers()[0]}")
    except Exception as e:
        print(f"✗ ERRO ao carregar modelo ONNX: {e}")
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
    
    # Inicializar variáveis de controle
    track_history = {}
    track_colors = {}
    global_max_area = 0.0
    last_approach_time = 0.0
    
    # Variáveis de estatísticas
    frame_count = 0
    total_processing_start_time = time.time()
    
    # Cache para segmentação
    last_horizon_mask = np.zeros((PROCESSING_HEIGHT, PROCESSING_WIDTH, 3), dtype=np.uint8)
    last_flight_status = "DESCONHECIDO"
    last_sky_ratio = 0.0
    last_roi_coords = (0, 0, 0, 0)
    last_status_color = (128, 128, 128)
    
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
        
        # ===== PROCESSAMENTO YOLO =====
        yolo_frame = resized_frame.copy()
        yolo_frame, approach_detected, global_max_area = process_yolo_detections(
            yolo_frame, yolo_model, TRACKER_CONFIG, YOLO_CONFIDENCE,
            track_history, track_colors, global_max_area
        )
        
        # Registrar hora do alerta se aproximação detectada
        if approach_detected:
            last_approach_time = time.time()
        
        # ===== PROCESSAMENTO DE SEGMENTAÇÃO (com skip frames) =====
        if (frame_count - 1) % SEGMENTATION_UPDATE_INTERVAL == 0:
            # Executar segmentação
            segmentation_mask_gray = run_onnx_inference(
                onnx_session,
                HORIZON_MODEL_INPUT_SIZE,
                resized_frame
            )
            
            # Binarizar máscara
            _, binary_mask = cv.threshold(segmentation_mask_gray, BINARY_THRESHOLD, 255, cv.THRESH_BINARY)
            
            # Analisar direção do voo
            last_flight_status, last_sky_ratio, last_roi_coords, last_status_color = analyze_flight_direction(binary_mask)
            
            # Converter máscara para BGR para visualização
            last_horizon_mask = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)
        
        # Desenhar informações de status do voo no frame de segmentação
        horizon_display = last_horizon_mask.copy()
        draw_flight_status(horizon_display, last_flight_status, last_sky_ratio, 
                          last_roi_coords, last_status_color)
        
        # ===== ADICIONAR INFORMAÇÕES VISUAIS =====
        # FPS
        frame_end_time = time.time()
        instant_fps = 1.0 / (frame_end_time - frame_start_time) if frame_end_time > frame_start_time else 0
        cv.putText(yolo_frame, f"FPS: {instant_fps:.1f}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame counter
        cv.putText(yolo_frame, f"Frame: {frame_count}/{total_frames}", (10, 55),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Desenhar alerta de aproximação
        draw_alert(yolo_frame, last_approach_time)
        
        # ===== COMBINAR E SALVAR FRAMES =====
        combined_frame = np.hstack((yolo_frame, horizon_display))
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