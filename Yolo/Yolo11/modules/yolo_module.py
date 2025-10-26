import torch
import os
import time
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import deque

# ============================= CLASSE YOLO DETECTOR =============================
class YOLODetector:
    """Classe responsável por detecção e tracking com YOLO"""
    
    def __init__(self, model_path, tracker_config, confidence_threshold, 
                 trail_length=50, approach_threshold=1.1, alert_duration=1.5,
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
        # Verificar GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA não disponível para YOLO")
        
        # Carregar modelo
        try:
            self.model = YOLO(model_path).to("cuda")
            print(f"✓ Modelo YOLO carregado na GPU")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo YOLO: {e}")
        
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.trail_length = trail_length
        
        # Histórico de tracking
        self.track_history = {}
        self.track_colors = {}
        
        # Sistema de alertas de aproximação
        self.global_max_area = 0.0
        self.last_approach_time = 0.0
        self.approach_area_threshold = approach_threshold
        self.alert_duration = alert_duration
        
        # Configurações visuais do alerta
        self.alert_message = alert_message
        self.alert_text_color = alert_text_color
        self.alert_box_color = alert_box_color
        self.alert_font_scale = alert_font_scale
        self.alert_thickness = alert_thickness
    
    def process_frame(self, frame):
        """
        Processa um frame com YOLO
        
        Args:
            frame: Frame BGR a ser processado
            
        Returns:
            tuple: (frame_processado, approach_detected)
        """
        frame_processed = frame.copy()
        
        # Executar YOLO com tracking
        results = self.model.track(
            frame_processed, 
            persist=True, 
            tracker=self.tracker_config,
            verbose=False, 
            conf=self.confidence_threshold
        )
        
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
                area = self._calculate_area(box)
                if area > current_frame_max_area:
                    current_frame_max_area = area
            
            # Verificar aproximação
            if self.global_max_area > 0 and current_frame_max_area > self.global_max_area * self.approach_area_threshold:
                approach_detected = True
                self.last_approach_time = time.time()
            
            # Atualizar recorde global
            self.global_max_area = max(self.global_max_area, current_frame_max_area)
            
            # Desenhar detecções e trilhas
            self._draw_detections(frame_processed, boxes, confidences, ids)
        
        # Desenhar alerta de aproximação se necessário
        self._draw_alert(frame_processed)
        
        return frame_processed, approach_detected
    
    def _calculate_area(self, box):
        """Calcula a área de uma caixa delimitadora"""
        x1, y1, x2, y2 = box
        return abs((x2 - x1) * (y2 - y1))
    
    def _draw_detections(self, frame, boxes, confidences, ids):
        """Desenha detecções e trilhas no frame"""
        for idx, (box, conf) in enumerate(zip(boxes, confidences)):
            tid = ids[idx] if idx < len(ids) else -1
            x1, y1, x2, y2 = box
            
            # Desenhar caixa
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adicionar label
            label_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
            cv.putText(frame, f" {conf:.2f}", label_pos,
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Atualizar histórico de tracking
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=self.trail_length)
                self.track_colors[tid] = (
                    int(np.random.randint(50, 255)),
                    int(np.random.randint(50, 255)),
                    int(np.random.randint(50, 255))
                )
            self.track_history[tid].append((cx, cy))
            
            # Desenhar trilha
            pts = np.array(self.track_history[tid], dtype=np.int32).reshape((-1, 1, 2))
            if len(pts) > 1:
                cv.polylines(frame, [pts], False, self.track_colors[tid], 2)
    
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