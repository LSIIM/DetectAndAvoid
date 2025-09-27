import torch
import os
import time
import cv2 as cv
import numpy as np
import onnxruntime
from ultralytics import YOLO
from collections import deque

# ===== CONFIGURAÇÕES =====
VIDEO_PATH = r"videos_test/droneVSdrone1.mp4"
YOLO_MODEL_PATH = r"Weights/yolo_11_JUNHO_nano_drones_DGX.engine"
HORIZON_MODEL_PATH = "Weights/skyseg_fp16.onnx"
TRACKER_CONFIG = "bytetrack.yaml"

input_dir, filename = os.path.split(VIDEO_PATH)
name, ext = os.path.splitext(filename)
OUTPUT_PATH = os.path.join(input_dir, f"{name}_unified{ext}")

YOLO_CONFIDENCE = 0.5
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 480
HORIZON_MODEL_INPUT_SIZE = (320, 320)

# --- Skip de segmentação (executa a cada N frames) ---
SEGMENTATION_UPDATE_INTERVAL = 20  

# Configurações do sistema de alerta
TRAIL_LENGTH = 50
APPROACH_AREA_INCREASE_THRESHOLD = 1.1  # 10% de aumento
ALERT_DURATION = 1.5                    # segundos
ALERT_MESSAGE = "# ALERTA: APROXIMACAO DETECTADA"
ALERT_TEXT_COLOR = (0, 0, 255)          # vermelho
ALERT_BOX_COLOR = (0, 0, 0)             # fundo preto
ALERT_FONT_SCALE = 1
ALERT_THICKNESS = 2

def calculate_area(box):
    x1, y1, x2, y2 = box
    return abs((x2 - x1) * (y2 - y1))

def run_onnx_inference(onnx_session, input_size_hw, image_bgr):
    original_height, original_width = image_bgr.shape[:2]

    resized_image = cv.resize(
        image_bgr,
        dsize=(input_size_hw[1], input_size_hw[0]),
        interpolation=cv.INTER_AREA
    )

    rgb_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
    normalized_image = np.array(rgb_image, dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_image = (normalized_image / 255.0 - mean) / std
    transposed_image = normalized_image.transpose(2, 0, 1)
    input_tensor = transposed_image.reshape(1, 3, input_size_hw[0], input_size_hw[1]).astype(np.float16)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    onnx_result = onnx_session.run([output_name], {input_name: input_tensor})
    output_mask = np.array(onnx_result).squeeze()

    min_val, max_val = np.min(output_mask), np.max(output_mask)
    if max_val > min_val:
        output_mask = (output_mask - min_val) / (max_val - min_val)
    else:
        output_mask = np.zeros_like(output_mask)

    output_mask_uint8 = (output_mask * 255).astype('uint8')

    return cv.resize(
        output_mask_uint8,
        (original_width, original_height),
        interpolation=cv.INTER_NEAREST
    )

def process_frame_for_horizon(frame, onnx_session, model_input_size_hw):
    segmentation_mask_gray = run_onnx_inference(onnx_session, model_input_size_hw, frame)
    mask_bgr = cv.cvtColor(segmentation_mask_gray, cv.COLOR_GRAY2BGR)
    return mask_bgr

# --- TensorRT/ORT session builder (usa TensorRT se disponível) ---
def build_session(model_path: str) -> onnxruntime.InferenceSession:
    os.makedirs("trt_cache", exist_ok=True)

    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    avail = onnxruntime.get_available_providers()
    providers = []

    if "TensorrtExecutionProvider" in avail:
        providers.append((
            "TensorrtExecutionProvider",
            {
                "trt_max_workspace_size": 1 << 30,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "trt_cache",
                "trt_builder_optimization_level": 5,
                "trt_timing_cache_enable": True,
                "trt_dla_enable": False,
            },
        ))
    if "CUDAExecutionProvider" in avail:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess = onnxruntime.InferenceSession(model_path, sess_options=so, providers=providers)
    print("SkySeg providers:", sess.get_providers())
    return sess


def main():
    print("--- Configurando Modelos ---")

    # Verificar e carregar modelo YOLO
    if not torch.cuda.is_available():
        print("ERRO: PyTorch não encontrou uma GPU compatível com CUDA. Verifique sua instalação.")
        print("O modelo YOLO não pode ser executado na GPU. Encerrando.")
        return
    
    try:
        yolo_model =  YOLO(YOLO_MODEL_PATH,task="segment")
        #yolo_model = YOLO(YOLO_MODEL_PATH).to("cuda")
        print(f"Modelo YOLO         : Carregado. Dispositivo: {yolo_model.device}")
    except Exception as e:
        print(f"ERRO ao carregar modelo YOLO na GPU: {e}")
        return

    # Carregar modelo ONNX
    try:
        onnx_session = build_session(HORIZON_MODEL_PATH)
        print(f"Modelo Horizonte(ONNX/TRT): Carregado. Provedores: {onnx_session.get_providers()}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Abrir vídeo
    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        return

    video_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_video = cv.VideoWriter(OUTPUT_PATH, fourcc, video_fps, (PROCESSING_WIDTH * 2, PROCESSING_HEIGHT))

    # Variáveis para tracking e alerta
    track_history = {}
    track_colors = {}
    global_max_area = 0.0
    last_approach_time = 0.0

    # Estatísticas
    frame_count = 0
    total_fps = 0.0
    total_processing_start_time = time.time()

    # Máscara cacheada para skip de segmentação
    last_horizon_mask = np.zeros((PROCESSING_HEIGHT, PROCESSING_WIDTH, 3), dtype=np.uint8)

    print("\n--- Iniciando Processamento ---")
    print(f"Vídeo de entrada: {os.path.basename(VIDEO_PATH)}")
    print(f"Vídeo de saída  : {os.path.basename(OUTPUT_PATH)}")
    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        frame_count += 1

        # Redimensionar frame
        resized_frame = cv.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT), interpolation=cv.INTER_AREA)

        # Frame para YOLO com tracking
        yolo_frame = resized_frame.copy()
        
        # Executar YOLO com tracking
        results = yolo_model.track(yolo_frame, persist=True, tracker=TRACKER_CONFIG, verbose=False, conf=YOLO_CONFIDENCE)
        
        approach_detected = False
        current_frame_max_area = 0.0

        # Processar detecções
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

            # Calcular área máxima deste frame
            for box in boxes:
                area = calculate_area(box)
                if area > current_frame_max_area:
                    current_frame_max_area = area

            # Verificar aproximação
            if global_max_area > 0 and current_frame_max_area > global_max_area * APPROACH_AREA_INCREASE_THRESHOLD:
                approach_detected = True

            # Atualizar recorde global
            global_max_area = max(global_max_area, current_frame_max_area)

            # Desenhar cada detecção e rastro
            for idx, (box, conf) in enumerate(zip(boxes, confidences)):
                tid = ids[idx] if idx < len(ids) else -1
                x1, y1, x2, y2 = box

                cv.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
                cv.putText(yolo_frame, f"{conf:.2f}", label_pos,
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if tid not in track_history:
                    track_history[tid] = deque(maxlen=TRAIL_LENGTH)
                    track_colors[tid] = (
                        int(np.random.randint(50, 255)),
                        int(np.random.randint(50, 255)),
                        int(np.random.randint(50, 255))
                    )
                track_history[tid].append((cx, cy))

                pts = np.array(track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                if len(pts) > 1:
                    cv.polylines(yolo_frame, [pts], False, track_colors[tid], 2)

        # Registra hora do alerta
        if approach_detected:
            last_approach_time = time.time()

        # --- Segmentação: atualizar somente a cada N frames ---
        if (frame_count - 1) % SEGMENTATION_UPDATE_INTERVAL == 0:
            last_horizon_mask = process_frame_for_horizon(
                resized_frame,
                onnx_session,
                HORIZON_MODEL_INPUT_SIZE
            )

        # Calcular FPS
        frame_end_time = time.time()
        instant_fps = 1.0 / (frame_end_time - frame_start_time) if frame_end_time > frame_start_time else 0
        total_fps += instant_fps
        cv.putText(yolo_frame, f"FPS: {instant_fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Exibe alerta dentro da janela
        if time.time() < last_approach_time + ALERT_DURATION:
            (tw, th), baseline = cv.getTextSize(ALERT_MESSAGE,
                                               cv.FONT_HERSHEY_SIMPLEX,
                                               ALERT_FONT_SCALE,
                                               ALERT_THICKNESS)
            pad = 5
            x1a, y1a = 15 - pad, 80 - th - pad
            x2a, y2a = 15 + tw + pad, 80 + baseline + pad
            cv.rectangle(yolo_frame, (x1a, y1a), (x2a, y2a), ALERT_BOX_COLOR, -1)
            cv.putText(yolo_frame, ALERT_MESSAGE, (15, 80),
                      cv.FONT_HERSHEY_SIMPLEX,
                      ALERT_FONT_SCALE,
                      ALERT_TEXT_COLOR,
                      ALERT_THICKNESS,
                      cv.LINE_AA)

        # Combinar frames lado a lado usando a última máscara calculada
        combined_frame = np.hstack((yolo_frame, last_horizon_mask))
        out_video.write(combined_frame)
        cv.imshow("YOLO Detection | Horizon Segmentation", combined_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Finalização
    total_processing_end_time = time.time()
    total_time = total_processing_end_time - total_processing_start_time
    average_fps = frame_count / total_time if total_time > 0 else 0
    
    cap.release()
    out_video.release()
    cv.destroyAllWindows()

    print("\n--- Processamento Concluído ---")
    print(f"Frames processados: {frame_count}")
    print(f"Tempo total: {total_time:.2f} segundos")
    print(f"FPS médio de processamento: {average_fps:.2f}")
    print(f"Vídeo de saída salvo em: {OUTPUT_PATH}")
    
if __name__ == '__main__':
    main()
