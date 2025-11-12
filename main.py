#!/usr/bin/env python3
"""
DetectAndAvoid - Main Integration Module

This module integrates YOLO detection, Sky Segmentation, and Optical Flow
processing into a unified video processing pipeline.

Usage:
    python main.py <video_path> [--clusters <num>] [--confidence <conf>]
"""

import argparse
import sys
from pathlib import Path
import torch
import os
import time
import cv2
import numpy as np
import onnxruntime
from ultralytics import YOLO
from collections import deque
from Yolo.Yolo11.modules.yolo_module import YOLODetector
from Yolo.Yolo11.modules.sky_seg_module import SkySegmentation
from OpticalFlow import opticalflow as optical_flow

YOLO_MODEL_PATH = r"Weights\yolo_11_JUNHO_nano_drones_DGX.pt"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DetectAndAvoid Integrated Processing")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for optical flow (default: 5)")
    parser.add_argument("--confidence", type=float, default=0.6, help="YOLO confidence threshold (default: 0.6)")
    parser.add_argument("--output", help="Output video path (optional)")
    parser.add_argument("--resize-height", type=int, default=480, help="Resize frame height (default: 480)")
    parser.add_argument("--yolo-model-path", type=str, default=YOLO_MODEL_PATH, help="Path to YOLO model weights")
    
    return parser.parse_args()

def setup_video_capture(video_path):
    """Setup video capture and get properties"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return cap, fps, frame_width, frame_height

def setup_video_writer(output_path, fps, width, height):
    """Setup video writer if output path is provided"""
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return None


# ============================= CONFIGURAÇÕES =============================
# Caminhos

HORIZON_MODEL_PATH = "Weights\skyseg_fp16.onnx"
USE_TENSORRT_SKYSEG = True

TRACKER_CONFIG = "bytetrack.yaml"

# Configurações de processamento
YOLO_CONFIDENCE = 0.5
HORIZON_MODEL_INPUT_SIZE = (320, 320)
SEGMENTATION_UPDATE_INTERVAL = 30  # Atualiza segmentação a cada N frames

# Configurações do sistema de alerta de aproximação
TRAIL_LENGTH = 50
APPROACH_AREA_INCREASE_THRESHOLD = 1.1  # 10% de aumento
ALERT_DURATION = 1.5  # segundos
ALERT_MESSAGE = "# ALERTA: APROXIMACAO DETECTADA"
ALERT_TEXT_COLOR = (0, 0, 255)  # vermelho
ALERT_BOX_COLOR = (0, 0, 0)  # fundo preto
ALERT_FONT_SCALE = 1
ALERT_THICKNESS = 2

# Configurações de análise de direção do voo
SAMPLE_AREA_SIZE = 30  # Tamanho da área de amostragem no centro
SKY_UPPER_THRESHOLD = 0.75  # Limiar para detectar SUBINDO
SKY_LOWER_THRESHOLD = 0.25  # Limiar para detectar DESCENDO
BINARY_THRESHOLD = 128  # Limiar para binarização da máscara

# Configuração de saída
input_dir, filename = os.path.split(VIDEO_PATH)
name, ext = os.path.splitext(filename)
OUTPUT_PATH = os.path.join(input_dir, f"{name}_unified_complete{ext}")



# ============================= FUNÇÃO PRINCIPAL =============================
def main():
    """Main integration function"""
    args = parse_arguments()
    
    print("=== DetectAndAvoid Integration System ===")
    print(f"Video: {args.video_path}")
    print(f"Clusters: {args.clusters}")
    print(f"YOLO Confidence: {args.confidence}")
    print("==========================================")
    
    # Setup video capture
    try:
        cap, fps, orig_width, orig_height = setup_video_capture(args.video_path)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Calculate resize scale
    resize_scale = args.resize_height / orig_height
    processing_width = int(orig_width * resize_scale)
    processing_height = args.resize_height
    
    print(f"Original resolution: {orig_width}x{orig_height}")
    print(f"Processing resolution: {processing_width}x{processing_height}")
    print(f"FPS: {fps}")
    
    # Setup video writer
    writer = setup_video_writer(args.output, fps, processing_width * 3, processing_height)
    
    # Setup modules
    print("\n--- Setting up modules ---")
    
    try:        
        # Optical Flow setup
        print("Setting up Optical Flow...")
        flow_context = optical_flow.setup(
            clusters=args.clusters, 
            fps=fps,
            processing_size=(processing_width, processing_height)
        )
        print("Setting up YOLO detector...")
        yolo_detector = YOLODetector(
            model_path=YOLO_MODEL_PATH,
            tracker_config=TRACKER_CONFIG,
            confidence_threshold=YOLO_CONFIDENCE,
            trail_length=TRAIL_LENGTH,
            approach_threshold=APPROACH_AREA_INCREASE_THRESHOLD,
            alert_duration=ALERT_DURATION,
            alert_message=ALERT_MESSAGE,
            alert_text_color=ALERT_TEXT_COLOR,
            alert_box_color=ALERT_BOX_COLOR,
            alert_font_scale=ALERT_FONT_SCALE,
            alert_thickness=ALERT_THICKNESS
        )
        print("Setting up Sky Segmentation...")
        sky_segmentation = SkySegmentation(
            model_path=HORIZON_MODEL_PATH,
            input_size=HORIZON_MODEL_INPUT_SIZE,
            update_interval=SEGMENTATION_UPDATE_INTERVAL,
            sample_area_size=SAMPLE_AREA_SIZE,
            sky_upper_threshold=SKY_UPPER_THRESHOLD,
            sky_lower_threshold=SKY_LOWER_THRESHOLD,
            binary_threshold=BINARY_THRESHOLD,
            use_tensorrt=USE_TENSORRT_SKYSEG 
        )
        
        print("All modules setup successfully!")
        
    except Exception as e:
        print(f"Error setting up modules: {e}")
        cap.release()
        if writer:
            writer.release()
        return 1
    
    # Main processing loop
    print("\n--- Starting video processing ---")
    frame_count = 0
    total_processing_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            #frame_start_time = time.time()
            frame_count += 1
        
            # Resize frame for processing
            resized_frame = cv2.resize(frame, (processing_width, processing_height))
            
            # Process frame with each module
            yolo_result, approach_detected = yolo_detector.process_frame(resized_frame)
            sky_result, flight_status, sky_ratio = sky_segmentation.process_frame(resized_frame)
            flow_result = optical_flow.process_frame(resized_frame, flow_context)
            
            # Create combined display
            combined_frame = np.hstack([yolo_result, sky_result, flow_result])
            
            # Add frame info
            info_text = f"Frame: {frame_count} | YOLO | Sky Seg | Optical Flow"
            cv2.putText(combined_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Write frame if output is specified
            if writer:
                writer.write(combined_frame)
            
            # Display results
            cv2.imshow("DetectAndAvoid - YOLO | Sky Segmentation | Optical Flow", combined_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):  # Save current frame
                cv2.imwrite(f"frame_{frame_count:06d}.jpg", combined_frame)
                print(f"Saved frame {frame_count}")
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
            
            # Atualizar progresso
            # if frame_count % 30 == 0:
            #     elapsed_time = time.time() - total_processing_start_time
            #     avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            #     eta = ((elapsed_time / frame_count) * (total_frames - frame_count)) if frame_count > 0 else 0
            #     progress = (frame_count / total_frames) * 100
            #     print(f"Progresso: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
            #           f"FPS médio: {avg_fps:.2f} | ETA: {eta:.1f}s")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Cleanup
        print(f"\n--- Processing completed ---")
        print(f"Total frames processed: {frame_count}")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Cleanup modules
        try:
            # yolo_detector.cleanup(yolo_context)
            # sky_segmentation.cleanup(sky_context)
            optical_flow.cleanup(flow_context)
        except:
            pass
    
    return 0


if __name__ == '__main__':
    sys.exit(main())