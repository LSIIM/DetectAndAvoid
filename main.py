#!/usr/bin/env python3
"""
DetectAndAvoid - Main Integration Module

This module integrates YOLO detection, ZipDepth, and Optical Flow
processing into a unified video processing pipeline.

Usage:
    python main.py <video_path> [--clusters <num>] [--confidence <conf>]
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from Yolo.Yolo11.modules.yolo_module import YOLODetector
# from Yolo.Yolo11.modules.sky_seg_module import SkySegmentation
from zip_depth.zip_depth_module import ZipDepth
from OpticalFlow import opticalflow as optical_flow

YOLO_MODEL_PATH = r"Yolo/Yolo11/Weights/best_yolo26_drone_bird_aircraft_junho_2026.engine"
# HORIZON_MODEL_PATH = r"Sky_Seg/skyseg_fp16.onnx"
ZIPDEPTH_ENGINE_PATH = r"zip_depth/zipdepth_base_384x384_fp16.trt"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DetectAndAvoid Integrated Processing")
    parser.add_argument("--video-ip", default="192.168.144.25", help="IP address for RTSP video stream")
    parser.add_argument("--video-path", type=str, help="Video Path for processing")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for optical flow (default: 5)")
    parser.add_argument("--confidence", type=float, default=0.6, help="YOLO confidence threshold (default: 0.6)")
    parser.add_argument("--output", help="Output video path (optional)")
    parser.add_argument("--resize-height", type=int, default=480, help="Resize frame height (default: 480)")
    parser.add_argument("--yolo-model-path", type=str, default=YOLO_MODEL_PATH, help="Path to YOLO model weights")
    # parser.add_argument("--horizon-model-path", type=str, default=HORIZON_MODEL_PATH, help="Path to Horizon model weights")
    # parser.add_argument("--segmentation-update-interval", type=int, default=30, help="Segmentation update interval (default: 30)")
    parser.add_argument("--depth-model-path", type=str, default=ZIPDEPTH_ENGINE_PATH, help="Path to ZipDepth TensorRT engine")
    parser.add_argument("--no-display", action="store_true", help="Skip cv2.imshow (keep --output if set)")
    parser.add_argument("--no-hw-decode", action="store_true", help="Force OpenCV software decode (no nvvidconv scale)")

    return parser.parse_args()


def gst_bgr_pipeline(uri, width=None, height=None, latency=None):
    """Jetson GStreamer pipeline: HW decode, optional nvvidconv scale, BGR appsink."""
    latency_attr = f" latency={latency}" if latency is not None else ""
    size = f",width={width},height={height}" if width and height else ""
    return (
        f"uridecodebin uri={uri}{latency_attr} ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        f"nvvidconv ! video/x-raw{size},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink sync=false max-buffers=1 drop=true"
    )


def _cap_props(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, frame_width, frame_height


def setup_video_capture_ip(ip, out_width=None, out_height=None):
    """Setup RTSP capture (GStreamer). Optional HW scale to out_width x out_height."""
    url = f"rtsp://{ip}:8554/main.264"
    cap = cv2.VideoCapture(gst_bgr_pipeline(url, out_width, out_height, latency=50), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {url}")
    fps, frame_width, frame_height = _cap_props(cap)
    return cap, fps, frame_width, frame_height


def setup_video_capture_path(path, hw_decode=True, out_width=None, out_height=None):
    """Setup file capture. HW decode+scale via nvvidconv when hw_decode and size given."""
    probe = cv2.VideoCapture(path)
    if not probe.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    fps, orig_width, orig_height = _cap_props(probe)

    if not hw_decode or not out_width or not out_height:
        return probe, fps, orig_width, orig_height, False

    probe.release()
    uri = "file://" + os.path.abspath(path)
    cap = cv2.VideoCapture(gst_bgr_pipeline(uri, out_width, out_height), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("GStreamer HW decode failed; falling back to OpenCV software decode")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")
        return cap, fps, orig_width, orig_height, False

    return cap, fps, orig_width, orig_height, True

def setup_video_writer(output_path, fps, width, height):
    """Setup video writer if output path is provided"""
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return None


# ============================= CONFIGURAÇÕES =============================
# Caminhos

# USE_TENSORRT_SKYSEG = True

TRACKER_CONFIG = "bytetrack.yaml"

# Configurações de processamento
YOLO_CONFIDENCE = 0.5
# HORIZON_MODEL_INPUT_SIZE = (320, 320)
# SEGMENTATION_UPDATE_INTERVAL = 30  # Atualiza segmentação a cada N frames

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
# SAMPLE_AREA_SIZE = 30  # Tamanho da área de amostragem no centro
# SKY_UPPER_THRESHOLD = 0.75  # Limiar para detectar SUBINDO
# SKY_LOWER_THRESHOLD = 0.25  # Limiar para detectar DESCENDO
# BINARY_THRESHOLD = 128  # Limiar para binarização da máscara


# ============================= FUNÇÕES DE PROCESSAMENTO PARALELO =============================
def process_yolo_threaded(frame, yolo_detector):
    """Process YOLO detection in a separate thread"""
    try:
        return yolo_detector.process_frame(frame)
    except Exception as e:
        print(f"Error in YOLO processing: {e}")
        return frame, False

# def process_sky_threaded(frame, sky_segmentation):
#     """Process sky segmentation in a separate thread"""
#     try:
#         return sky_segmentation.process_frame(frame)
#     except Exception as e:
#         print(f"Error in Sky Segmentation processing: {e}")
#         return frame, "UNKNOWN", 0.0

def process_depth_threaded(frame, zip_depth):
    """Process ZipDepth in a separate thread"""
    try:
        return zip_depth.process_frame(frame)
    except Exception as e:
        print(f"Error in ZipDepth processing: {e}")
        return np.zeros_like(frame)

def process_flow_threaded(frame, flow_context):
    """Process optical flow in a separate thread"""
    try:
        return optical_flow.process_frame(frame, flow_context)
    except Exception as e:
        print(f"Error in Optical Flow processing: {e}")
        return frame

# ============================= FUNÇÃO PRINCIPAL =============================
def main():
    """Main integration function"""
    args = parse_arguments()
    
    print("=== DetectAndAvoid Integration System ===")
    if not args.video_path:
        print(f"Video IP: {args.video_ip}")
    else:
        print(f"Video Path: {args.video_path}")        
    print(f"Clusters: {args.clusters}")
    print(f"YOLO Confidence: {args.confidence}")
    print("==========================================")
    
    hw_scaled = False
    try:
        if not args.video_path:
            url = f"rtsp://{args.video_ip}:8554/main.264"
            probe = cv2.VideoCapture(gst_bgr_pipeline(url, latency=50), cv2.CAP_GSTREAMER)
            if not probe.isOpened():
                raise ValueError(f"Could not open video file: {url}")
            fps, orig_width, orig_height = _cap_props(probe)
            probe.release()
            processing_width = int(orig_width * (args.resize_height / orig_height))
            processing_height = args.resize_height
            cap, fps, cap_w, cap_h = setup_video_capture_ip(
                args.video_ip, processing_width, processing_height
            )
            if cap_w > 0 and cap_h > 0:
                processing_width, processing_height = cap_w, cap_h
                hw_scaled = True
        else:
            probe = cv2.VideoCapture(args.video_path)
            if not probe.isOpened():
                raise ValueError(f"Could not open video file: {args.video_path}")
            fps, orig_width, orig_height = _cap_props(probe)
            probe.release()
            processing_width = int(orig_width * (args.resize_height / orig_height))
            processing_height = args.resize_height
            cap, fps, _ow, _oh, hw_scaled = setup_video_capture_path(
                args.video_path,
                hw_decode=not args.no_hw_decode,
                out_width=processing_width,
                out_height=processing_height,
            )
            if hw_scaled:
                cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
                ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
                if cw > 0 and ch > 0:
                    processing_width, processing_height = cw, ch
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"Original resolution: {orig_width}x{orig_height}")
    print(f"Processing resolution: {processing_width}x{processing_height}")
    print(f"HW decode+scale: {hw_scaled}")
    print(f"Display: {'off' if args.no_display else 'on'}")
    print(f"FPS: {fps}")
    
    # Setup video writer (side-by-side: YOLO+flow | ZipDepth)
    writer = setup_video_writer(args.output, fps, processing_width * 2, processing_height)
    # red_overlay = np.full_like(np.zeros((processing_height, processing_width, 3), dtype=np.uint8), (0, 0, 127))

    
    # Setup modules
    print("\n--- Setting up modules ---")
    
    try:        
        # Optical Flow setup
        print("Setting up Optical Flow...")
        flow_context = optical_flow.setup(
            max_point=40
        )
        print("Setting up YOLO detector...")
        yolo_detector = YOLODetector(
            model_path=args.yolo_model_path,
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
        print("Setting up ZipDepth...")
        zip_depth = ZipDepth(
            model_path=args.depth_model_path
        )
        # print("Setting up Sky Segmentation...")
        # sky_segmentation = SkySegmentation(
        #     model_path=args.horizon_model_path,
        #     input_size=HORIZON_MODEL_INPUT_SIZE,
        #     update_interval=args.segmentation_update_interval,
        #     sample_area_size=SAMPLE_AREA_SIZE,
        #     sky_upper_threshold=SKY_UPPER_THRESHOLD,
        #     sky_lower_threshold=SKY_LOWER_THRESHOLD,
        #     binary_threshold=BINARY_THRESHOLD,
        #     use_tensorrt=USE_TENSORRT_SKYSEG 
        # )
        
        print("All modules setup successfully!")
        
    except Exception as e:
        print(f"Error setting up modules: {e}")
        cap.release()
        if writer:
            writer.release()
        return 1
    
    # Setup thread pool for parallel processing
    # Use 3 threads for 3 modules (optimal for Jetson Orin NX with 8 cores)
    executor = ThreadPoolExecutor(max_workers=3)
    
    # Main processing loop
    print("\n--- Starting video processing ---")
    print("Using parallel processing with 3 threads")
    frame_count = 0
    total_processing_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            frame_count += 1
        
            # Frame already at processing size when nvvidconv scaled
            if hw_scaled:
                resized_frame = frame
            else:
                resized_frame = cv2.resize(frame, (processing_width, processing_height))
            
            # Shared buffer: workers do not write the color frame
            future_yolo = executor.submit(process_yolo_threaded, resized_frame, yolo_detector)
            # future_sky = executor.submit(process_sky_threaded, resized_frame.copy(), sky_segmentation)
            future_depth = executor.submit(process_depth_threaded, resized_frame, zip_depth)
            future_flow = executor.submit(process_flow_threaded, resized_frame, flow_context)
            
            # Wait for all results (parallel execution happens here)
            yolo_result, yolo_confidence, yolo_ids, yolo_approach_detected = future_yolo.result()
            # sky_result, sky_flight_status, sky_ratio = future_sky.result()
            depth_color = future_depth.result()
            flow_new, flow_ids, flow_uvs = future_flow.result()
            
            frame_processing_time = time.time() - frame_start_time
            
            # Create combined display
            combined_frame = resized_frame.copy()

            # sky_result in 50% alpha red in combined_frame
            # if sky_result is not None:
            #     alpha = 0.5
            #     colored_region = cv2.addWeighted(combined_frame, 1 - alpha, red_overlay, alpha, 0)
            #     combined_frame[sky_result == 255] = colored_region[sky_result == 255]

            # Draw yolo_result detections on combined_frame
            if yolo_result is not None:
                combined_frame = yolo_detector.draw_detections(combined_frame, yolo_result, yolo_confidence, yolo_ids)

            # Draw optical flow on combined_frame
            vetor = [0,0]
            for i, pid in enumerate(flow_ids) if flow_new is not None else []:
                new = flow_new[i]
                vetor += flow_uvs[i]
                a, b = int(new[0]), int(new[1])
                u, v = flow_uvs[i] * fps

                # Draw arrow for optical flow
                combined_frame = cv2.circle(combined_frame, (a, b), 5, flow_context.colors[0], -1)
                combined_frame = cv2.arrowedLine(combined_frame, (a, b), (int(a + u), int(b + v)), flow_context.colors[1], 2, tipLength=0.2)

            if  flow_new is not None:
                vetor/len(flow_ids)
                vetor * fps

            combined_frame = cv2.circle(combined_frame, (int(processing_width/2), int(processing_height/2)), 8, (40,40,40), -1)
            combined_frame = cv2.arrowedLine(combined_frame, (int(processing_width/2), int(processing_height/2)), (int(processing_width/2 + vetor[0]), int(processing_height/2 + vetor[1])), (80,120,80), 3, tipLength=0.2)

            # Add frame info with processing time
            info_text = f"Frame: {frame_count} | YOLO | ZipDepth | Optical Flow | {frame_processing_time*1000:.1f}ms"
            cv2.putText(combined_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            display = np.hstack([combined_frame, depth_color])

            # Write frame if output is specified
            if writer:
                writer.write(display)
            
            if not args.no_display:
                cv2.imshow("DetectAndAvoid - YOLO | Optical Flow | ZipDepth", display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"frame_{frame_count:06d}.jpg", display)
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
        # Shutdown thread pool
        executor.shutdown(wait=True)
        
        # Cleanup
        total_time = time.time() - total_processing_start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n--- Processing completed ---")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Cleanup modules
        try:
            optical_flow.cleanup(flow_context)
        except:
            pass
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
