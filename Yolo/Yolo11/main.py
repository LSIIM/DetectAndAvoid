import cv2
import time
from ultralytics import YOLO
import argparse

DEFAULT_VIDEO_PATH = r"Raw_Videos\fev_corte_3.mp4"
MODEL_PATH = r"Weights\best_fev_2025.pt"
DEFAULT_OUTPUT_PATH = r"detection_optimized.mp4"
CONFIDENCE_THRESHOLD = 0.6

def main(video_source, output_path):
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_source}")
        return

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) if video_source != 0 else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, video_fps, (video_width, video_height))

    if not out_video.isOpened():
        print(f"Erro ao criar o arquivo de vídeo de saída: {output_path}")
        cap.release()
        return

    frame_counter = 0
    total_processing_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        start_time = time.time()

        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

        if results and results[0].boxes:
            for detection_data in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, _ = detection_data
                x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1_int, y1_int), (x2_int, y2_int), (0, 0, 255), 2)
                conf_text = f"{conf:.2f}"
                text_y_pos = y1_int - 10 if y1_int > 10 else y1_int + 15
                cv2.putText(frame, conf_text, (x1_int, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        current_loop_time = time.time() - start_time
        current_fps = 1.0 / current_loop_time if current_loop_time > 0 else 0
        total_processing_fps += current_fps

        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out_video.write(frame)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_fps = total_processing_fps / frame_counter if frame_counter > 0 else 0

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    print(f"Processamento concluído.")
    print(f"Vídeo de detecção salvo em: {output_path}")
    print(f"Total de frames processados: {frame_counter}")
    print(f"FPS médio de processamento: {avg_fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa vídeo com YOLO para detecção de objetos.")
    parser.add_argument('--camera', action='store_true', help='Usar a câmera do dispositivo (cv2.VideoCapture(0)) como fonte de vídeo.')
    args = parser.parse_args()

    video_input_source = 0 if args.camera else DEFAULT_VIDEO_PATH
    output_file_path = "camera_detection.mp4" if args.camera else DEFAULT_OUTPUT_PATH

    main(video_input_source, output_file_path)