import torch
import os
import time
import cv2 as cv
import numpy as np
import onnxruntime

class SkySegmentation:
    """Classe responsável por segmentação do céu e análise de direção de voo"""
    
    def __init__(self, model_path, input_size=(320, 320), update_interval=30,
                 sample_area_size=30, sky_upper_threshold=0.75, 
                 sky_lower_threshold=0.25, binary_threshold=128):
        """
        Inicializa o segmentador de céu
        
        Args:
            model_path: Caminho para o modelo ONNX
            input_size: Tamanho de entrada do modelo (altura, largura)
            update_interval: Intervalo de frames para atualizar segmentação
            sample_area_size: Tamanho da área de amostragem no centro
            sky_upper_threshold: Limiar para detectar SUBINDO (ex: 0.75 = 75% céu)
            sky_lower_threshold: Limiar para detectar DESCENDO (ex: 0.25 = 25% céu)
            binary_threshold: Limiar para binarização da máscara (0-255)
        """
        # Carregar modelo
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            print(f"✓ Modelo de segmentação carregado: {self.session.get_providers()[0]}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo ONNX: {e}")
        
        self.input_size = input_size
        self.update_interval = update_interval
        
        # Cache para otimização
        self.frame_count = 0
        self.last_mask = None
        self.last_flight_status = "DESCONHECIDO"
        self.last_sky_ratio = 0.0
        self.last_roi_coords = (0, 0, 0, 0)
        self.last_status_color = (128, 128, 128)
        
        # Parâmetros de análise
        self.sample_area_size = sample_area_size
        self.sky_upper_threshold = sky_upper_threshold
        self.sky_lower_threshold = sky_lower_threshold
        self.binary_threshold = binary_threshold
    
    def process_frame(self, frame):
        """
        Processa um frame para segmentação de céu
        
        Args:
            frame: Frame BGR a ser processado
            
        Returns:
            tuple: (frame_visualizacao, status_voo, sky_ratio)
        """
        self.frame_count += 1
        
        # Verificar se precisa atualizar segmentação
        should_update = (self.frame_count - 1) % self.update_interval == 0
        
        if should_update or self.last_mask is None:
            # Executar inferência
            mask_gray = self._run_inference(frame)
            
            # Binarizar máscara
            _, binary_mask = cv.threshold(mask_gray, self.binary_threshold, 255, cv.THRESH_BINARY)
            
            # Analisar direção do voo
            (self.last_flight_status, 
             self.last_sky_ratio, 
             self.last_roi_coords, 
             self.last_status_color) = self._analyze_flight_direction(binary_mask)
            
            # Salvar máscara processada
            self.last_mask = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)
        
        # Criar frame de visualização
        display_frame = self.last_mask.copy()
        self._draw_flight_status(display_frame)
        
        return display_frame, self.last_flight_status, self.last_sky_ratio
    
    def _run_inference(self, image_bgr):
        """Executa inferência do modelo ONNX"""
        original_height, original_width = image_bgr.shape[:2]
        
        # Redimensionar e preparar imagem
        resized_image = cv.resize(
            image_bgr,
            dsize=(self.input_size[1], self.input_size[0]),
            interpolation=cv.INTER_AREA
        )
        
        # Converter para RGB e normalizar
        rgb_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        normalized_image = np.array(rgb_image, dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized_image = (normalized_image / 255.0 - mean) / std
        
        # Transpor e preparar tensor
        transposed_image = normalized_image.transpose(2, 0, 1)
        input_tensor = transposed_image.reshape(1, 3, self.input_size[0], self.input_size[1]).astype(np.float16)
        
        # Executar inferência
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        onnx_result = self.session.run([output_name], {input_name: input_tensor})
        output_mask = np.array(onnx_result).squeeze()
        
        # Normalizar saída
        min_val, max_val = np.min(output_mask), np.max(output_mask)
        if max_val > min_val:
            output_mask = (output_mask - min_val) / (max_val - min_val)
        else:
            output_mask = np.zeros_like(output_mask)
        
        output_mask_uint8 = (output_mask * 255).astype('uint8')
        
        # Redimensionar para tamanho original
        return cv.resize(
            output_mask_uint8,
            (original_width, original_height),
            interpolation=cv.INTER_NEAREST
        )
    
    def _analyze_flight_direction(self, binary_mask):
        """
        Analisa a direção do voo baseado na proporção de céu no centro
        
        Returns:
            tuple: (status, sky_ratio, roi_coords, color)
        """
        height, width = binary_mask.shape[:2]
        center_y, center_x = height // 2, width // 2
        half_size = self.sample_area_size // 2
        
        # Definir região de interesse (ROI)
        y_start = max(0, center_y - half_size)
        y_end = min(height, center_y + half_size)
        x_start = max(0, center_x - half_size)
        x_end = min(width, center_x + half_size)
        
        # Calcular proporção de céu na ROI
        center_roi = binary_mask[y_start:y_end, x_start:x_end]
        sky_ratio = np.mean(center_roi) / 255.0
        
        # Determinar status do voo
        if sky_ratio > self.sky_upper_threshold:
            status = "SUBINDO"
            color = (0, 255, 255)  # amarelo
        elif sky_ratio < self.sky_lower_threshold:
            status = "DESCENDO"
            color = (255, 0, 0)    # azul
        else:
            status = "NIVELADO"
            color = (0, 255, 0)    # verde
        
        return status, sky_ratio, (x_start, y_start, x_end, y_end), color
    
    def _draw_flight_status(self, frame):
        """Desenha informações de status do voo no frame"""
        x_start, y_start, x_end, y_end = self.last_roi_coords
        
        # Desenhar ROI
        cv.rectangle(frame, (x_start, y_start), (x_end, y_end), self.last_status_color, 3)
        
        # Desenhar cruz central
        self._draw_center_cross(frame, size=15, color=self.last_status_color, thickness=2)
        
        # Adicionar texto com status
        status_text = f"VOO: {self.last_flight_status}"
        ratio_text = f"CEU: {self.last_sky_ratio:.1%}"
        
        # Posicionar textos no canto superior direito
        text_x = frame.shape[1] - 200
        cv.putText(frame, status_text, (text_x, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, self.last_status_color, 2)
        cv.putText(frame, ratio_text, (text_x, 55),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, self.last_status_color, 2)
    
    def _draw_center_cross(self, image, size=20, color=(0, 0, 255), thickness=2):
        """Desenha uma cruz no centro da imagem"""
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        cv.line(image, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
        cv.line(image, (center_x, center_y - size), (center_x, center_y + size), color, thickness)
    
    def reset(self):
        """Reseta o estado do segmentador"""
        self.frame_count = 0
        self.last_mask = None
        self.last_flight_status = "DESCONHECIDO"
        self.last_sky_ratio = 0.0
        self.last_roi_coords = (0, 0, 0, 0)
        self.last_status_color = (128, 128, 128)