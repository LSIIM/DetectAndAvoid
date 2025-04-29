import cv2
import numpy as np
import time
from ultralytics import YOLO

video_path = r"Raw_Videos/fev_corte_2.mp4"
model_path = r"best_fev_2025.engine" 
output_path = r"detection_tensorrt_output.mp4" 

CONFIDENCE_THRESHOLD = 0.3

print(f"Carregando modelo TensorRT de: {model_path}")
try:
    model = YOLO(model_path) 
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifique se o arquivo .engine existe no caminho especificado e se é compatível com seu ambiente (GPU, CUDA, TensorRT).")
    exit()

print("Modelo carregado com sucesso.")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    print(f"Erro ao criar o arquivo de vídeo de saída: {output_path}")
    cap.release()
    exit()

frame_count = 0
total_fps = 0
print("Iniciando processamento do vídeo...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura do frame.")
        break

    frame_count += 1
    start_time = time.time()

    results = model(frame, verbose=False) 
    
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for i, box in enumerate(boxes):
            if confidences[i] >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                conf_text = f"{confidences[i]:.2f}"
                cv2.putText(frame, conf_text, (x1, y1 - 10 if y1 > 10 else y1 + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    elapsed_time = time.time() - start_time
    fps_val = 1.0 / elapsed_time if elapsed_time > 0 else 0
    total_fps += fps_val
    
    cv2.putText(frame, f"FPS: {fps_val:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow("Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Processamento interrompido pelo usuário.")
        break

average_fps = total_fps / frame_count if frame_count > 0 else 0

print("Finalizando...")
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processamento concluído.")
print(f"Vídeo de detecção salvo em: {output_path}")
print(f"Total de frames processados: {frame_count}")
print(f"FPS médio: {average_fps:.2f}")