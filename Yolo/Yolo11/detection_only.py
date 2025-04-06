import cv2
import numpy as np
import time
from ultralytics import YOLO

video_path = "Raw_Videos/fev_corte_3.mp4"
model = YOLO("best.pt")

CONFIDENCE_THRESHOLD = 0.5

output_path = "detection_only.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
total_fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    start_time = time.time()

    results = model(frame)
    
    if len(results) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for i, box in enumerate(boxes):
            if confidences[i] >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf_text = f"{confidences[i]:.2f}"
                cv2.putText(frame, conf_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elapsed_time = time.time() - start_time
    fps_val = 1.0 / elapsed_time if elapsed_time > 0 else 0
    total_fps += fps_val
    
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

average_fps = total_fps / frame_count if frame_count > 0 else 0

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vídeo de detecção salvo em: {output_path}")
print(f"FPS médio: {average_fps:.2f}")