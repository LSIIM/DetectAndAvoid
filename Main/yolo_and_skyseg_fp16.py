import torch
import os
import copy
import time
import cv2 as cv
import numpy as np
import onnxruntime
from ultralytics import YOLO

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

def process_frame_for_horizon(frame, onnx_session, model_input_size_hw, prev_horizon_smoothed=None):
    window_width_percentage = 0.20
    max_vertical_change_per_frame = 10
    interpolation_factor = 0.15
    
    segmentation_mask_gray = run_onnx_inference(onnx_session, model_input_size_hw, frame)

    threshold = 128
    H, W = segmentation_mask_gray.shape

    horizon_points = []
    for col in range(W):
        col_data = segmentation_mask_gray[:, col]
        transition_row = H - 1
        for row in range(1, H):
            if col_data[row] < threshold and col_data[row-1] >= threshold:
                transition_row = row
                break
        horizon_points.append(transition_row)

    window_size = int(W * window_width_percentage)
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(15, window_size)

    current_horizon_smoothed = []
    half_window = window_size // 2
    for i in range(W):
        start, end = max(0, i - half_window), min(W, i + half_window + 1)
        window_values = horizon_points[start:end]
        smoothed_val = int(np.median(window_values)) if window_values else H - 1
        current_horizon_smoothed.append(smoothed_val)
    
    current_horizon_smoothed = np.array(current_horizon_smoothed)

    if prev_horizon_smoothed is not None and len(prev_horizon_smoothed) == len(current_horizon_smoothed):
        delta = current_horizon_smoothed - prev_horizon_smoothed
        clipped_delta = np.clip(delta, -max_vertical_change_per_frame, max_vertical_change_per_frame)
        constrained_target_horizon = prev_horizon_smoothed + clipped_delta
        
        final_horizon = (
            (1 - interpolation_factor) * prev_horizon_smoothed + 
            interpolation_factor * constrained_target_horizon
        ).astype(np.int32)
    else:
        final_horizon = current_horizon_smoothed

    mask_with_line_bgr = cv.cvtColor(segmentation_mask_gray, cv.COLOR_GRAY2BGR)

    line_color, line_thickness = (0, 0, 255), 2
    for col in range(W - 1):
        pt1 = (col, final_horizon[col])
        pt2 = (col + 1, final_horizon[col+1])
        if 0 <= pt1[1] < H and 0 <= pt2[1] < H:
            cv.line(mask_with_line_bgr, pt1, pt2, line_color, line_thickness)

    return mask_with_line_bgr, final_horizon

def main():
    YOLO_MODEL_PATH = r"Weights\best_fev_2025.pt"
    HORIZON_MODEL_PATH = "skyseg_fp16.onnx"
    VIDEO_PATH = r"videos_test\droneVSdrone1.mp4"
    
    input_dir, filename = os.path.split(VIDEO_PATH)
    name, ext = os.path.splitext(filename)
    OUTPUT_PATH = os.path.join(input_dir, f"{name}__fp16SkySeg_combinado{ext}")

    YOLO_CONFIDENCE = 0.6
    PROCESSING_WIDTH = 640
    PROCESSING_HEIGHT = 480
    HORIZON_MODEL_INPUT_SIZE = (320, 320)

    print("--- Configurando Modelos ---")

    
    if not torch.cuda.is_available():
        print("ERRO: PyTorch não encontrou uma GPU compatível com CUDA. Verifique sua instalação.")
        print("O modelo YOLO não pode ser executado na GPU. Encerrando.")
        return
    
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH).to("cuda")
        #yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"Modelo YOLO         : Carregado. Dispositivo: {yolo_model.device}")
    except Exception as e:
        print(f"ERRO ao carregar modelo YOLO na GPU: {e}")
        return

    

    try:
        onnx_session = onnxruntime.InferenceSession(HORIZON_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"Modelo Horizonte(ONNX): Carregado. Provedor: {onnx_session.get_providers()}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        return

    video_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_video = cv.VideoWriter(OUTPUT_PATH, fourcc, video_fps, (PROCESSING_WIDTH * 2, PROCESSING_HEIGHT))

    print("\n--- Iniciando Processamento ---")
    print(f"Vídeo de entrada: {os.path.basename(VIDEO_PATH)}")
    print(f"Vídeo de saída  : {os.path.basename(OUTPUT_PATH)}")
    
    total_processing_start_time = time.time()
    frame_count = 0
    previous_horizon = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        frame_count += 1

        resized_frame = cv.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT), interpolation=cv.INTER_AREA)

        yolo_frame = resized_frame.copy()
        #yolo_results = yolo_model(yolo_frame, verbose=False, conf=YOLO_CONFIDENCE)
        yolo_results = yolo_model(yolo_frame, device='cuda', verbose=False, conf=YOLO_CONFIDENCE)
        if yolo_results and yolo_results[0].boxes:
            for *xyxy, conf, cls in yolo_results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                cv.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label_y = y1 - 10 if y1 > 10 else y1 + 15
                cv.putText(yolo_frame, f"{conf:.2f}", (x1, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        horizon_mask_frame, current_horizon = process_frame_for_horizon(
            resized_frame,
            onnx_session,
            HORIZON_MODEL_INPUT_SIZE,
            prev_horizon_smoothed=previous_horizon
        )
        previous_horizon = current_horizon

        frame_end_time = time.time()
        instant_fps = 1.0 / (frame_end_time - frame_start_time) if frame_end_time > frame_start_time else 0
        cv.putText(yolo_frame, f"FPS: {instant_fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined_frame = np.hstack((yolo_frame, horizon_mask_frame))
        out_video.write(combined_frame)
        cv.imshow("YOLO Detection | Horizon Segmentation", combined_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

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