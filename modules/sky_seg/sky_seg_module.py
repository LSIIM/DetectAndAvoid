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
                 sky_lower_threshold=0.25, binary_threshold=128, use_tensorrt=True):
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

        try:
            self.session = self._build_session(model_path, use_tensorrt)
            print(f"✓ Modelo de segmentação carregado: {self.session.get_providers()[0]}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo ONNX: {e}")
        
        self.input_size = input_size
        self.update_interval = update_interval
        

        self.frame_count = 0
        self.last_mask = None
        self.last_flight_status = "DESCONHECIDO"
        self.last_sky_ratio = 0.0
        self.last_roi_coords = (0, 0, 0, 0)
        self.last_status_color = (128, 128, 128)
        

        self.sample_area_size = sample_area_size
        self.sky_upper_threshold = sky_upper_threshold
        self.sky_lower_threshold = sky_lower_threshold
        self.binary_threshold = binary_threshold
    
    def _build_session(self, model_path, use_tensorrt):
        """Constrói sessão ONNX com TensorRT se disponível"""
        os.makedirs("trt_cache", exist_ok=True)
        
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        avail = onnxruntime.get_available_providers()
        providers = []
        
        if use_tensorrt and "TensorrtExecutionProvider" in avail:  
            os.makedirs("trt_cache", exist_ok=True)
            providers.append((
                "TensorrtExecutionProvider",
                {
                    "trt_max_workspace_size": 1 << 30,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "trt_cache",
                    "trt_builder_optimization_level": 5,
                    "trt_timing_cache_enable": True,
                    "trt_dla_enable": False,
                },
            ))
        
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        return onnxruntime.InferenceSession(model_path, sess_options=so, providers=providers)
    
    
    def process_frame(self, frame):
        """
        Processa um frame para segmentação de céu
        
        Args:
            frame: Frame BGR a ser processado
            
        Returns:
            tuple: (frame_visualizacao, status_voo, sky_ratio)
        """
        self.frame_count += 1
        
        
        should_update = (self.frame_count - 1) % self.update_interval == 0
        
        if should_update or self.last_mask is None:
            mask_gray = self._run_inference(frame)

            _, binary_mask = cv.threshold(mask_gray, self.binary_threshold, 255, cv.THRESH_BINARY)
            

            (self.last_flight_status, 
             self.last_sky_ratio, 
             self.last_roi_coords, 
             self.last_status_color) = self._analyze_flight_direction(binary_mask)

            self.last_mask = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)

        display_frame = self.last_mask.copy()
        #self._draw_flight_status(display_frame)
        
        return display_frame, self.last_flight_status, self.last_sky_ratio
    
    def _run_inference(self, image_bgr):
        """Executa inferência do modelo ONNX"""
        original_height, original_width = image_bgr.shape[:2]

        resized_image = cv.resize(
            image_bgr,
            dsize=(self.input_size[1], self.input_size[0]),
            interpolation=cv.INTER_AREA
        )
        
        rgb_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        normalized_image = np.array(rgb_image, dtype=np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized_image = (normalized_image / 255.0 - mean) / std
        
        transposed_image = normalized_image.transpose(2, 0, 1)
        input_tensor = transposed_image.reshape(1, 3, self.input_size[0], self.input_size[1]).astype(np.float16)
        

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        onnx_result = self.session.run([output_name], {input_name: input_tensor})
        output_mask = np.array(onnx_result).squeeze()
        

        min_val, max_val = np.min(output_mask), np.max(output_mask)
        if max_val > min_val:
            output_mask = (output_mask - min_val) / (max_val - min_val)
        else:
            output_mask = np.zeros_like(output_mask)
        
        output_mask_uint8 = (output_mask * 255).astype('uint8')
        
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
        

        y_start = max(0, center_y - half_size)
        y_end = min(height, center_y + half_size)
        x_start = max(0, center_x - half_size)
        x_end = min(width, center_x + half_size)
        
        center_roi = binary_mask[y_start:y_end, x_start:x_end]
        sky_ratio = np.mean(center_roi) / 255.0
        

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
        

        cv.rectangle(frame, (x_start, y_start), (x_end, y_end), self.last_status_color, 3)

        self._draw_center_cross(frame, size=15, color=self.last_status_color, thickness=2)

        status_text = f"VOO: {self.last_flight_status}"
        ratio_text = f"CEU: {self.last_sky_ratio:.1%}"

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

if __name__ == "__main__":
    model_path = "models/sky_segmentation.onnx"
    sky_seg = SkySegmentation(model_path=model_path, input_size=(320, 320), update_interval=30)

    cap = cv.VideoCapture(0)  # Captura da webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame, status, sky_ratio = sky_seg.process_frame(frame)

        cv.imshow("Sky Segmentation", display_frame)
        print(f"Status: {status}, Sky Ratio: {sky_ratio:.2%}")

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()