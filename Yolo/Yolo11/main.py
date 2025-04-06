import cv2
import numpy as np
import time
from ultralytics import YOLO

# Caminho do vídeo e do modelo (ajuste conforme necessário)
video_path = "Raw_Videos/fev_corte_3.mp4"
model = YOLO("best.pt")

# Parâmetros iniciais
MAX_TRAJECTORY_LENGTH = 90

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

tracker = None
trajectory = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    start_time = time.time()

    # A cada 30 frames ou se não houver tracker ativo, utiliza YOLO para detecção
    if tracker is None or frame_count % 30 == 0:
        results = model(frame)
        if len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            if boxes.shape[0] > 0:
                # Seleciona a primeira detecção
                x1, y1, x2, y2 = boxes[0][:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                initBB = (x1, y1, x2 - x1, y2 - y1)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, initBB)

                # Desenha o bounding box e atualiza a trajetória
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                trajectory.append((center_x, center_y))
                trajectory = trajectory[-MAX_TRAJECTORY_LENGTH:]
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
    else:
        # Atualiza o tracker com o frame atual
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            trajectory.append((center_x, center_y))
            trajectory = trajectory[-MAX_TRAJECTORY_LENGTH:]
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        else:
            tracker = None

    # Desenha a trajetória (linhas conectando os pontos)
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

    # Calcula e exibe os FPS do frame atual
    elapsed_time = time.time() - start_time
    fps_val = 1.0 / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Exibe o frame com a detecção e rastreamento
    cv2.imshow("Tracking", frame)
    
    # Encerra se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
