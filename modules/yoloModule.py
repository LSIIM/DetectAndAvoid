import torch
import os
import time
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import deque

class YOLODetector:
    """Classe responsável por detecção e tracking com YOLO integrada ao pipeline de Fluxo Óptico"""
    
    def __init__(self, modelPath, trackerConfig, confidenceThreshold, 
                 trailLength = 50):
        
        try:
            _, ext = os.path.splitext(modelPath)
            
            if ext.lower() == '.engine':
                self.model = YOLO(modelPath, task="segment")
                print(f"Modelo YOLO TensorRT carregado (.engine)")

            elif ext.lower() == '.pt':
                self.model = YOLO(modelPath)
                print(f"Modelo YOLO PyTorch carregado (.pt)")

            else:
                raise ValueError(f"Formato de modelo não suportado: {ext}. Use .pt ou .engine")
                
        except Exception as e:

            raise RuntimeError(f"Erro ao carregar modelo YOLO: {e}")
        
        self.trackerConfig = trackerConfig
        self.confidenceThreshold = confidenceThreshold
        self.trailLength = trailLength
        
        self.trackHistory = {}
        self.trackColors = {}
    
    def processFrame(self, frame):
        """
        Processa um frame com YOLO
        
        Args:
            frame: Frame BGR a ser processado
            
        Returns:
            boxes
        """
        results = self.model.track(

            frame, 
            persist=True, 
            tracker=self.trackerConfig,
            verbose=False, 
            conf=self.confidenceThreshold
        )
        
        boxes = np.array([])
        
        if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # if len(boxes) > 0:

            #     if results[0].boxes.conf is not None:
            #         confidences = results[0].boxes.conf.cpu().numpy()

            #     else:
            #         confidences = np.zeros(len(boxes), dtype=float)
                
            #     if results[0].boxes.id is not None:
            #         IDs = results[0].boxes.id.int().cpu().tolist()

            #     else:
            #         IDs = [-1] * len(boxes)

            #     self._drawDetections(frame, boxes.astype(int), confidences, IDs)

        return boxes
    
    def _drawDetections(self, frame, boxes, confidences, IDs):

        """Desenha detecções e trilhas no frame"""

        for IDx, (box, conf) in enumerate(zip(boxes, confidences)):

            tID = IDs[IDx] if IDx < len(IDs) and IDs[IDx] is not None else -1
            x1, y1, x2, y2 = box

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            labelPos = (x1, y1 - 10 if y1 > 20 else y1 + 20)
            cv.putText(frame, f"{conf:.2f}", labelPos, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if tID != -1:
                if tID not in self.trackHistory:

                    self.trackHistory[tID] = deque(maxlen=self.trailLength)
                    self.trackColors[tID] = (
                        int(np.random.randint(50, 255)),
                        int(np.random.randint(50, 255)),
                        int(np.random.randint(50, 255))
                    )

                self.trackHistory[tID].append((cx, cy))
                
                pts = np.array(self.trackHistory[tID], dtype=np.int32).reshape((-1, 1, 2))
                
                if len(pts) > 1:
                    cv.polylines(frame, [pts], False, self.trackColors[tID], 2)

    def reset(self):
        
        self.trackHistory.clear()
        self.trackColors.clear()