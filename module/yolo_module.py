import torch
import os
import time
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import deque

class kalman_filter:
    """Classe auxiliar para encapsular o filtro de Kalman"""
    
    def __init__(self, process_noise=1e-2, measurement_noise=1e-1, prediction_horizon_sec=1.5):
        """
        prediction_horizon_sec: Horizonte da predição futura em segundos
        process_noise: Ruído do modelo de movimento do filtro
        measurement_noise: Ruído da medição do YOLO
        """
        self.kalman_filters = {}
        self.kalman_last_update = {}
        self.predicted_positions = {}
        self.prediction_horizon_sec = prediction_horizon_sec
        self.kalman_process_noise = process_noise
        self.kalman_measurement_noise = measurement_noise

    def _create_kalman_filter(self, center):
        """Cria um filtro com estado [x, y, vx, vy] para uma detecção."""
        kalman = cv.KalmanFilter(4, 2, 0, cv.CV_32F)
        kalman.transitionMatrix = np.eye(4, dtype=np.float32)
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * self.kalman_process_noise
        kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * self.kalman_measurement_noise
        )
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman.statePost = np.array(
            [[center[0]], [center[1]], [0], [0]], dtype=np.float32
        )
        return kalman

    def _update_kalman_filter(self, track_id, center, timestamp):
        """Atualiza o filtro de um objeto com o centro observado pelo YOLO."""
        kalman = self.kalman_filters.get(track_id)
        if kalman is None:
            self.kalman_filters[track_id] = self._create_kalman_filter(center)
            self.kalman_last_update[track_id] = timestamp
            return

        dt = max(1e-3, min(timestamp - self.kalman_last_update[track_id], 1.0))
        kalman.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32
        )
        kalman.predict()
        kalman.correct(np.array([[center[0]], [center[1]]], dtype=np.float32))
        self.kalman_last_update[track_id] = timestamp

    def _predict_future_position(self, track_id, horizon_sec):
        """Retorna (x, y) do centro previsto após o horizonte informado."""
        kalman = self.kalman_filters[track_id]
        future_matrix = kalman.transitionMatrix.copy()
        future_matrix[0, 2] = horizon_sec
        future_matrix[1, 3] = horizon_sec
        state = kalman.statePost
        future_position = future_matrix @ state
        return int(round(float(future_position[0, 0]))), int(round(float(future_position[1, 0])))

    def predict_future_positions(self, horizon_sec=None):
        """Prevê a posição futura de cada track conhecido pelo filtro de Kalman."""
        horizon = (
            self.prediction_horizon_sec
            if horizon_sec is None else max(0.0, horizon_sec)
        )
        return {
            track_id: self._predict_future_position(track_id, horizon)
            for track_id in self.predicted_positions
        }

    def process_kalman(self, ids, boxes,timestamp):
        if len(ids) > 0 and len(boxes) > 0:
            self.predicted_positions = {}
        future_center = []
        for box, track_id in zip(boxes, ids):
            center = self._box_center(box)
            self._update_kalman_filter(track_id, center, timestamp)
            self.predicted_positions[track_id] = self._predict_future_position(
                track_id, self.prediction_horizon_sec
            )
            future_center.append(self.predicted_positions[track_id])
        return future_center

    def draw_predicted_positions(self, frame):
        """Desenha as posições previstas no frame"""
        for track_id, future_center in self.predicted_positions.items():
            cv.circle(frame, future_center, 5, (0, 165, 255), -1)
            cv.putText(frame, "pred", (future_center[0] + 6, future_center[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        return frame
    
    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = box
        return (float(x1 + x2) / 2, float(y1 + y2) / 2)
    
    def reset(self):
        """Reseta o estado do filtro de Kalman"""
        self.kalman_filters.clear()
        self.kalman_last_update.clear()
        self.predicted_positions.clear()


class YOLODetector:
    """Classe responsável por detecção e tracking com YOLO"""
    
    def __init__(self, model_path, tracker_config, confidence_threshold, 
                 trail_length=50, approach_threshold=1.1, alert_duration=1.5,
                 no_det_reset_sec=1.5,
                 alert_message="# ALERTA: APROXIMACAO DETECTADA",
                 alert_text_color=(0, 0, 255), alert_box_color=(0, 0, 0),
                 alert_font_scale=1, alert_thickness=2):
        """
        Inicializa o detector YOLO
        
        Args:
            model_path: Caminho para o modelo YOLO
            tracker_config: Arquivo de configuração do tracker
            confidence_threshold: Limiar de confiança para detecções
            trail_length: Comprimento da trilha de tracking
            approach_threshold: Threshold para detectar aproximação (ex: 1.1 = 10% aumento)
            alert_duration: Duração do alerta em segundos
            alert_message: Mensagem do alerta
            alert_text_color: Cor do texto do alerta (B, G, R)
            alert_box_color: Cor do fundo do alerta (B, G, R)
            alert_font_scale: Escala da fonte do alerta
            alert_thickness: Espessura do texto do alerta
        """
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("CUDA não disponível; executando YOLO .pt na CPU")
        
        try:
            _, ext = os.path.splitext(model_path)
            
            if ext.lower() == '.engine':
                if self.device == "cpu":
                    raise RuntimeError(
                        "Modelo TensorRT (.engine) requer CUDA; use um modelo .pt para CPU"
                    )
                self.model = self._load_engine(model_path)
                print("✓ Modelo YOLO TensorRT carregado (.engine)")
            elif ext.lower() == '.pt':
                self.model = YOLO(model_path).to(self.device)
                print(f"✓ Modelo YOLO PyTorch carregado em {self.device} (.pt)")
            else:
                raise ValueError(f"Formato não suportado: {ext}")
                
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo YOLO: {e}")
        
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.trail_length = trail_length
        
        self.track_history = {}
        self.track_colors = {}

        self.global_max_area = 0.0
        self.last_approach_time = 0.0
        self.last_detection_time = 0.0
        self.approach_area_threshold = approach_threshold
        self.alert_duration = alert_duration
        self.no_det_reset_sec = no_det_reset_sec

        self.alert_message = alert_message
        self.alert_text_color = alert_text_color
        self.alert_box_color = alert_box_color
        self.alert_font_scale = alert_font_scale
        self.alert_thickness = alert_thickness
    
    def _load_engine(self, model_path: str) -> YOLO:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for task in ("segment","detect"):
            try:
                model = YOLO(model_path, task=task)
                model(dummy, verbose=False)
                print(f"  task inferida: {task}")
                return model
            except (IndexError, Exception):
                continue
        raise RuntimeError(f"Não foi possível inferir task para: {model_path}")

    def process_frame(self, frame):
        """
        Processa um frame com YOLO
        
        Args:
            frame: Frame BGR a ser processado
            
        Returns:
            tuple: (boxes, confidences, ids, approach_detected)
        """
        frame_processed = frame.copy()
        
        results = self.model.track(
            frame_processed, 
            persist=True, 
            tracker=self.tracker_config,
            verbose=False, 
            conf=self.confidence_threshold,
            device=self.device
        )
        
        approach_detected = False
        current_frame_max_area = 0.0
        now = time.time()
        boxes = np.empty((0, 4), dtype=int)
        confidences = np.empty(0, dtype=float)
        ids = []
        
        has_detection = False
        if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            if len(boxes) > 0:
                has_detection = True
                self.last_detection_time = now
            
            if results[0].boxes.conf is not None:
                confidences = results[0].boxes.conf.cpu().numpy()
            else:
                confidences = np.zeros(len(boxes), dtype=float)
            
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
            else:
                ids = list(range(len(boxes)))

            for box in boxes:
                area = self._calculate_area(box)
                if area > current_frame_max_area:
                    current_frame_max_area = area

            if self.global_max_area > 0 and current_frame_max_area > self.global_max_area * self.approach_area_threshold:
                approach_detected = True
                self.last_approach_time = time.time()

            self.global_max_area = max(self.global_max_area, current_frame_max_area)

            #self._draw_detections(frame_processed, boxes, confidences, ids)

        if not has_detection:
            if self.last_detection_time > 0 and (now - self.last_detection_time) > self.no_det_reset_sec:
                self.global_max_area = 0.0

        #self._draw_alert(frame_processed)
        
        return boxes, confidences, ids, approach_detected
    
    def _calculate_area(self, box):
        """Calcula a área de uma caixa delimitadora"""
        x1, y1, x2, y2 = box
        return abs((x2 - x1) * (y2 - y1))
    
    def draw_detections(self, frame, boxes, confidences, ids):
        """Desenha detecções e trilhas no frame"""
        future_center = None
        w, h = 0, 0
        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            tid = ids[idx] if idx < len(ids) else -1
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
            cv.putText(frame, f" {conf:.2f}", label_pos,
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=self.trail_length)
                self.track_colors[tid] = (
                    int(np.random.randint(50, 255)),
                    int(np.random.randint(50, 255)),
                    int(np.random.randint(50, 255))
                )
            self.track_history[tid].append((cx, cy))
            
            pts = np.array(self.track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) > 1:
                cv.polylines(frame, [pts], False, self.track_colors[tid], 2)
        return frame, future_center, w, h
    
    def _draw_alert(self, frame):
        """Desenha alerta de aproximação se necessário"""
        if time.time() < self.last_approach_time + self.alert_duration:
            (tw, th), baseline = cv.getTextSize(
                self.alert_message,
                cv.FONT_HERSHEY_SIMPLEX,
                self.alert_font_scale,
                self.alert_thickness
            )
            pad = 5
            x1a, y1a = 15 - pad, 80 - th - pad
            x2a, y2a = 15 + tw + pad, 80 + baseline + pad
            
            cv.rectangle(frame, (x1a, y1a), (x2a, y2a), self.alert_box_color, -1)
            cv.putText(frame, self.alert_message, (15, 80),
                      cv.FONT_HERSHEY_SIMPLEX,
                      self.alert_font_scale,
                      self.alert_text_color,
                      self.alert_thickness,
                      cv.LINE_AA)
    
    def reset(self):
        """Reseta o estado do detector"""
        self.track_history.clear()
        self.track_colors.clear()
        self.global_max_area = 0.0
        self.last_approach_time = 0.0


if __name__ == "__main__":
    # Exemplo de uso do YOLODetector
    model_path = "Yolo/Yolo11/Weights/best_yolo26_drone_bird_aircraft_junho_2026.pt"  # Substitua pelo caminho do seu modelo
    tracker_config = "bytetrack.yaml"  # Substitua pelo caminho do seu arquivo de configuração do tracker
    confidence_threshold = 0.5

    detector = YOLODetector(model_path, tracker_config, confidence_threshold)
    kalman = kalman_filter(process_noise=1e-2, measurement_noise=1e-1, prediction_horizon_sec=0.5)

    # create .log file to print confidences
    log_file_path = "conf.txt"
    with open(log_file_path, "w") as log_file:
        log_file.write("")


    cap = cv.VideoCapture("Videos/droneVSdrone1.mp4")  # Captura da webcam
    height, width = cap.get(cv.CAP_PROP_FRAME_HEIGHT), cap.get(cv.CAP_PROP_FRAME_WIDTH)
    new_height = 640
    new_width = int(width * (new_height / height))
    # fps = cap.get(cv.CAP_PROP_FPS)
    # writer = cv.VideoWriter("CUT_REC_Yolo.mp4", cv.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
    min_y, max_y, min_x, max_x = 0, new_height, 0, new_width
    frame_n = 0
    future_center = None
    last_center = None
    uv = None
    while True:
        ret, frame = cap.read()
        frame_n += 1
        if not ret:
            break

        #Resize to 640 height, maintaining aspect ratio
        frame = cv.resize(frame, (new_width, new_height))
        # yolo_frame = frame[min_y:max_y, min_x:max_x]
        boxes, confidences, ids, approach_detected = detector.process_frame(frame)
        # boxes = boxes + np.array([min_x, min_y, min_x, min_y])  # Ajusta as coordenadas para o frame original
        
        if len(boxes) > 0:
            tmp_uv = (np.array(boxes[0][0:2]) + np.array(boxes[0][2:4])) / 2
            uv = tmp_uv - np.array(last_center) if last_center is not None else np.array([0, 0])
            last_center = tmp_uv
        else:
            last_center = None
            uv = None

        if uv is not None:
            # Log 
            with open(log_file_path, "a") as log_file:
                log_file.write(f"({uv[0]},{uv[1]})\n")
        else:
            with open(log_file_path, "a") as log_file:
                log_file.write("\n")

        last_future_center = future_center
        future_center = kalman.process_kalman(ids, [kalman._box_center(box) for box in boxes], time.time())          
        # future_center = kalman.predicted_positions[ids[0]] if ids else None 

        if future_center is not None and last_future_center is not None and False:
            uv = np.array(future_center) - np.array(last_future_center)
            v = np.linalg.norm(uv)
            if v > 100:
                kalman.prediction_horizon_sec = 0.5
            elif v > 10:
                kalman.prediction_horizon_sec = 1.5
            else:
                kalman.prediction_horizon_sec = 2

        # kalman.draw_predicted_positions(frame)
        f, _, w, h = detector.draw_detections(frame, boxes, confidences, ids)
        
        if future_center is not None and not(future_center[0] < 0 or future_center[1] < 0) and False:
            max_x, max_y = new_width, new_height
            min_x, min_y = 0, 0
            multi = 10
            
            max_x = max(min(max_x, future_center[0] + w * int(multi/1)), 20)
            min_x = min(max(min_x, future_center[0] - w * int(multi/1)), new_width - 20)
            max_y = max(min(max_y, future_center[1] + h * multi), 20)
            min_y = min(max(min_y, future_center[1] - h * multi), new_height - 20)
            

            crop = frame[min_y:max_y, min_x:max_x]
            frame[0:crop.shape[0], 0:crop.shape[1]] = crop
            cv.rectangle(frame, (0, 0), (crop.shape[1] + 2, crop.shape[0] + 2), (0, 0, 200), 2)
        else:
            min_y, max_y, min_x, max_x = 0, new_height, 0, new_width
        
        cv.imshow("YOLO Detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        # writer.write(frame)

    cap.release()
    # writer.release()
    cv.destroyAllWindows()
