import cv2
import time
import threading
from ultralytics import YOLO


video_path   = r"Raw_Videos/fev_corte_3.mp4"
output_path  = r"Real_time_fev_corte_3_processado.mp4" # salva video
model        = YOLO(r"Weights/best_fev_2025.pt")#.to("cpu")


def simulate_camera_from_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro: não foi possível abrir o vídeo.")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay  = 1.0 / fps
    w, h         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    writer       = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    stop_threads       = False
    processing         = False
    frames_dropped     = 0
    frames_processed   = 0
    current_frame      = None
    frame_lock         = threading.Lock()

    def capture_frames():
        nonlocal stop_threads, processing, frames_dropped
        while not stop_threads:
            ret, frame = cap.read()
            if not ret:
                stop_threads = True
                break

            if processing:
                frames_dropped += 1
            else:
                threading.Thread(
                    target=process_frame_thread,
                    args=(frame.copy(),),
                    daemon=True,
                ).start()

            time.sleep(frame_delay)

    def process_frame_thread(frame):
        nonlocal processing, frames_processed, current_frame
        processing = True
        try:
            results = model(frame, conf=0.6)
            if results:
                boxes       = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{confidences[i]:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2)

            with frame_lock:
                current_frame = frame.copy()

            writer.write(frame)         
            frames_processed += 1
        finally:
            processing = False

    def display_frames():
        nonlocal stop_threads, current_frame
        cv2.namedWindow("Camera Simulation", cv2.WINDOW_NORMAL)
        while not stop_threads:
            with frame_lock:
                if current_frame is not None:
                    view = current_frame.copy()
                    status = "PROCESSANDO..." if processing else "AGUARDANDO"
                    color  = (0, 165, 255) if processing else (0, 255, 0)
                    cv2.putText(view, status, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.imshow("Camera Simulation", view)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                stop_threads = True
                break
            time.sleep(0.033)

    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=display_frames, daemon=True).start()

    try:
        while not stop_threads:
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_threads = True
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print("\nEstatísticas finais:")
        print(f"Frames processados: {frames_processed}")
        print(f"Frames dropados:    {frames_dropped}")

simulate_camera_from_video(video_path, output_path)
