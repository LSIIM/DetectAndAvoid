# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import deque

#VIDEO_PATH = r"Raw_Videos\fev_corte_2.mp4"
VIDEO_PATH = r"Raw_Videos/fev_corte_2.mp4"
OUTPUT_PATH = r"alert_system_yolo_SMALL_TENSOR_RT.mp4"
#MODEL_PATH = r"Weights/best_yolo11_small_abril.pt"
MODEL_PATH = r"Weights/best_yolo11_small_abril.engine"


TRAIL_LENGTH = 50
TRACKER_CONFIG = "bytetrack.yaml"
CONFIANCA = 0.5



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

def main():
    
    model = YOLO(MODEL_PATH, task="segment")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {VIDEO_PATH}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, input_fps, (width, height))
    if not out.isOpened():
        print(f"Error: Cannot create output file: {OUTPUT_PATH}")
        cap.release()
        return

    # Histórico de rastros
    track_history = {}
    track_colors  = {}

    global_max_area    = 0.0
    last_approach_time = 0.0

    frame_count = 0
    total_fps   = 0.0

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start_time = time.time()

        results = model.track(frame, persist=True, tracker=TRACKER_CONFIG, verbose=False,conf=CONFIANCA)

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

            
            for box in boxes:
                area = calculate_area(box)
                if area > current_frame_max_area:
                    current_frame_max_area = area

            
            if global_max_area > 0 and current_frame_max_area > global_max_area * APPROACH_AREA_INCREASE_THRESHOLD:
                approach_detected = True

            
            global_max_area = max(global_max_area, current_frame_max_area)

            
            for idx, (box, conf) in enumerate(zip(boxes, confidences)):
                tid = ids[idx] if idx < len(ids) else -1
                x1, y1, x2, y2 = box

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
                cv2.putText(frame, f"{conf:.2f}", label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if tid not in track_history:
                    track_history[tid] = deque(maxlen=TRAIL_LENGTH)
                    track_colors[tid]  = (
                        int(np.random.randint(50, 255)),
                        int(np.random.randint(50, 255)),
                        int(np.random.randint(50, 255))
                    )
                track_history[tid].append((cx, cy))

                pts = np.array(track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, track_colors[tid], 2)

        
        if approach_detected:
            last_approach_time = time.time()

        # Calcula e exibe FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        total_fps += fps
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
        if time.time() < last_approach_time + ALERT_DURATION:
            (tw, th), baseline = cv2.getTextSize(ALERT_MESSAGE,
                                                 cv2.FONT_HERSHEY_SIMPLEX,
                                                 ALERT_FONT_SCALE,
                                                 ALERT_THICKNESS)
            pad = 5
            x1a, y1a = 15 - pad, 80 - th - pad
            x2a, y2a = 15 + tw + pad, 80 + baseline + pad
            cv2.rectangle(frame, (x1a, y1a), (x2a, y2a), ALERT_BOX_COLOR, -1)
            cv2.putText(frame, ALERT_MESSAGE, (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        ALERT_FONT_SCALE,
                        ALERT_TEXT_COLOR,
                        ALERT_THICKNESS,
                        cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Tracking + Alerta", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_fps = total_fps / frame_count if frame_count else 0
    print("Finished.")
    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
