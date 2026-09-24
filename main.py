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
import queue
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from modules.YOLO.yolo_module import YOLODetector
from modules.depth.zip_depth_module import ZipDepth
from modules.Optical_Flow import opticalflow as optical_flow

YOLO_MODEL_PATH = r"Yolo/Yolo11/Weights/best_yolo26_drone_bird_aircraft_junho_2026.engine"
ZIPDEPTH_ENGINE_PATH = r"weights/zipdepth_base_384x384_fp16.trt"


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
    parser.add_argument("--depth-model-path", type=str, default=ZIPDEPTH_ENGINE_PATH, help="Path to ZipDepth TensorRT engine")
    parser.add_argument("--no-display", action="store_true", help="Skip cv2.imshow (keep --output if set)")

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


def setup_video_capture_path(path, out_width=None, out_height=None, hw_decode=True):
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

def _align32(n):
    """nvv4l2h264enc requires width/height multiple of 32 on Jetson."""
    return max(32, (int(n) + 31) // 32 * 32)


class GstHwVideoWriter:
    """HW encode in a separate gst-launch process (nvv4l2h264enc).

    NVDEC (OpenCV GStreamer capture) and NVENC cannot run together on this
    Jetson — the decoder hits EOS after ~2 frames even if encode is
    out-of-process. File capture therefore uses FFmpeg while this writer runs.
    """

    def __init__(self, output_path, fps, width, height, bitrate=8_000_000):
        self.enc_w, self.enc_h = _align32(width), _align32(height)
        self._pad = None
        fps_n = max(1, int(round(fps or 30)))
        nvmm = (
            f"video/x-raw(memory:NVMM),format=NV12,"
            f"width={self.enc_w},height={self.enc_h}"
        )
        cmd = [
            "gst-launch-1.0", "-e", "-q",
            "fdsrc", "!",
            "rawvideoparse",
            f"width={self.enc_w}",
            f"height={self.enc_h}",
            "format=bgr",
            f"framerate={fps_n}/1",
            "!",
            "videoconvert", "!", "video/x-raw,format=BGRx", "!",
            "nvvidconv", "!", nvmm, "!",
            "nvv4l2h264enc",
            f"bitrate={bitrate}",
            "insert-sps-pps=true",
            f"idrinterval={fps_n}",
            "preset-level=1",
            "maxperf-enable=1",
            "!",
            "h264parse", "!", "qtmux", "!",
            "filesink", f"location={output_path}",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )
        time.sleep(0.2)
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"gst-launch-1.0 exited immediately (code {self.proc.returncode}). "
                "Confirm gst-launch-1.0, rawvideoparse, nvvidconv and nvv4l2h264enc."
            )

    def isOpened(self):
        return self.proc is not None and self.proc.poll() is None

    def write(self, frame):
        if not self.isOpened() or self.proc.stdin is None:
            raise RuntimeError("GStreamer HW writer is not running")
        img = np.ascontiguousarray(frame)
        if img.shape[0] != self.enc_h or img.shape[1] != self.enc_w:
            if self._pad is None:
                self._pad = np.zeros((self.enc_h, self.enc_w, 3), dtype=np.uint8)
            h = min(img.shape[0], self.enc_h)
            w = min(img.shape[1], self.enc_w)
            self._pad[:h, :w] = img[:h, :w]
            if h < self.enc_h:
                self._pad[h:, :] = 0
            if w < self.enc_w:
                self._pad[:, w:] = 0
            img = self._pad
        self.proc.stdin.write(img.tobytes())

    def release(self):
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
        self.proc = None


def setup_video_writer(output_path, fps, width, height, bitrate=8_000_000):
    """Setup GStreamer HW encoder (nvv4l2h264enc) if output path is provided."""
    if not output_path:
        return None

    try:
        writer = GstHwVideoWriter(output_path, fps, width, height, bitrate)
    except FileNotFoundError:
        print("WARNING: gst-launch-1.0 not found. Recording disabled.")
        return None
    except Exception as e:
        print(f"WARNING: GStreamer HW writer failed to start: {e} Recording disabled.")
        return None

    print(
        f"Video writer: gst-launch nvv4l2h264enc "
        f"in {width}x{height} -> enc {writer.enc_w}x{writer.enc_h} -> {output_path}"
    )
    return writer


# ============================= CONFIGURAÇÕES =============================
# Caminhos

TRACKER_CONFIG = "bytetrack.yaml"

# Configurações de processamento
YOLO_CONFIDENCE = 0.5

# Configurações do sistema de alerta de aproximação
TRAIL_LENGTH = 50
APPROACH_AREA_INCREASE_THRESHOLD = 1.1  # 10% de aumento
ALERT_DURATION = 1.5  # segundos
ALERT_MESSAGE = "# ALERTA: APROXIMACAO DETECTADA"
ALERT_TEXT_COLOR = (0, 0, 255)  # vermelho
ALERT_BOX_COLOR = (0, 0, 0)  # fundo preto
ALERT_FONT_SCALE = 1
ALERT_THICKNESS = 2


# ============================= FUNÇÕES DE PROCESSAMENTO PARALELO =============================
def process_yolo_threaded(frame, yolo_detector):
    """Process YOLO detection in a separate thread"""
    try:
        return yolo_detector.process_frame(frame)
    except Exception as e:
        print(f"Error in YOLO processing: {e}")
        return frame, False

def process_depth_threaded(frame, zip_depth):
    """Process ZipDepth in a separate thread"""
    try:
        return zip_depth.process_frame(frame)
    except Exception as e:
        print(f"Error in ZipDepth processing: {e}")
        return np.zeros_like(frame)

def process_flow_threaded(frame, flow_context):
    try:
        return optical_flow.process_frame(frame, flow_context)  
    except Exception as e:
        print(f"Error in Optical Flow processing: {e}")
        return frame, False   


def _print_breakdown(
    times_capture,
    times_resize,
    times_queue_wait,
    times_yolo,
    times_depth,
    times_flow,
    times_combine,
    times_write,
    zip_depth=None,
    flow_context=None,
    yolo_detector=None,
):
    def _line(label, xs):
        if not xs:
            return
        print(f"{label:22s} {np.mean(xs) * 1000:6.2f} ms/frame")

    print("\n--- Breakdown médio por etapa ---")
    _line("Capture:", times_capture)
    _line("Resize (CPU shared):", times_resize)
    _line("Queue wait:", times_queue_wait)
    _line("YOLO (wait result):", times_yolo)
    _line("Depth (wait result):", times_depth)
    _line("Flow (wait result):", times_flow)
    _line("Combine:", times_combine)
    _line("Write:", times_write)
    if yolo_detector is not None:
        _line("YOLO preprocess:", getattr(yolo_detector, "_times_preprocess", None))
        _line("YOLO inference:", getattr(yolo_detector, "_times_inference", None))
        _line("YOLO postprocess:", getattr(yolo_detector, "_times_postprocess", None))
        _line("YOLO track() total:", getattr(yolo_detector, "_times_track_total", None))
        _line("YOLO post loop:", getattr(yolo_detector, "_times_post_loop", None))
    if zip_depth is not None:
        _line("Depth infer:", getattr(zip_depth, "_times_infer", None))
        _line("Depth postprocess:", getattr(zip_depth, "_times_postprocess", None))
    if flow_context is not None:
        _line("Flow LK:", getattr(flow_context, "_times_lk", None))
        _line("Flow rest:", getattr(flow_context, "_times_rest", None))


def min_wind(roi, min_size, max_width, max_height):
    """Ensure the ROI has a minimum size and is within bounds"""
    x1, y1, x2, y2 = roi
    if x2 - x1 < min_size:
        x2 = x2 + min_size//2
        if x2 > max_width:
            x1 -= ((max_width - x2) + min_size//2)
            x2 = max_width
        else:
            x1 -= min_size//2
            if x1 < 0:
                x1 = 0
                x2 = min_size
    if y2 - y1 < min_size:
        y2 = y2 + min_size//2
        if y2 > max_height:
            y1 -= ((max_height - y2) + min_size//2)
            y2 = max_height
        else:
            y1 -= min_size//2
            if y1 < 0:
                y1 = 0
                y2 = min_size
    return [x1, y1, x2, y2]

def boxes_roi(boxes, min_size, max_width, max_height):
    """Build one bounded ROI containing all tracked detections."""
    if boxes is None or len(boxes) == 0:
        return None
    boxes = np.asarray(boxes)
    return min_wind(
        [
            int(np.min(boxes[:, 0])),
            int(np.min(boxes[:, 1])),
            int(np.max(boxes[:, 2])),
            int(np.max(boxes[:, 3])),
        ],
        min_size,
        max_width,
        max_height,
    )

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
            processing_width = int(orig_width * (args.resize_height / orig_height)) & ~1
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
            processing_width = int(orig_width * (args.resize_height / orig_height)) & ~1
            processing_height = args.resize_height
            # NVDEC + NVENC together EOS the capture after ~2 frames on this Jetson.
            hw_decode = not bool(args.output)
            cap, fps, _ow, _oh, hw_scaled = setup_video_capture_path(
                args.video_path,
                out_width=processing_width,
                out_height=processing_height,
                hw_decode=hw_decode,
            )
            if not hw_decode:
                print("File capture: OpenCV/FFmpeg (NVDEC off while HW encoder is active)")
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

    
    # Setup modules
    print("\n--- Setting up modules ---")
    
    try:        
        # Optical Flow setup
        print("Setting up Optical Flow...")
        flow_context = optical_flow.setup(
            max_point=40,
            number_clusters=args.clusters
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
        
        print("All modules setup successfully!")
        
    except Exception as e:
        print(f"Error setting up modules: {e}")
        cap.release()
        if writer:
            writer.release()
        return 1
    
    # Use 3 threads for 3 modules (optimal for Jetson Orin NX with 8 cores)
    executor = ThreadPoolExecutor(max_workers=3)
    
    # Main processing loop
    print("\n--- Starting video processing ---")
    print("Using parallel processing with 3 threads + capture producer")
    frame_count = 0
    total_processing_start_time = time.time()
    times_capture = []
    times_resize = []
    times_queue_wait = []
    times_yolo = []
    times_depth = []
    times_flow = []
    times_combine = []
    times_write = []

    SENTINEL = None
    frame_q = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    def _put_frame(item):
        while True:
            if item is not SENTINEL and stop_event.is_set():
                return False
            try:
                frame_q.put(item, timeout=0.2)
                return True
            except queue.Full:
                if item is SENTINEL:
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    continue
                if stop_event.is_set():
                    return False
                continue

    def capture_worker():
        frame_id = 0
        try:
            while not stop_event.is_set():
                t0 = time.time()
                ret, frame = cap.read()
                capture_dt = time.time() - t0
                if not ret:
                    print(f"Capture ended (ret=False) after {frame_id} frames")
                    break
                times_capture.append(capture_dt)
                if hw_scaled:
                    resized_frame = frame
                    times_resize.append(0.0)
                else:
                    t0 = time.time()
                    resized_frame = cv2.resize(frame, (processing_width, processing_height))
                    times_resize.append(time.time() - t0)
                frame_id += 1
                if not _put_frame((frame_id, resized_frame)):
                    break
        finally:
            _put_frame(SENTINEL)

    capture_thread = threading.Thread(target=capture_worker, name="capture", daemon=True)
    capture_thread.start()
    
    try:
        while True:
            t0 = time.time()
            item = frame_q.get()
            times_queue_wait.append(time.time() - t0)
            if item is SENTINEL:
                break
            frame_count, resized_frame = item
            frame_start_time = time.time()
            
<<<<<<< HEAD
            # Shared buffer: workers do not write the color frame
            future_yolo = executor.submit(process_yolo_threaded, resized_frame, yolo_detector)
            future_depth = executor.submit(process_depth_threaded, resized_frame, zip_depth)
            future_flow = executor.submit(process_flow_threaded, resized_frame, flow_context)
=======
            # Submit all processing tasks in parallel
            future_yolo = executor.submit(process_yolo_threaded, resized_frame.copy(), yolo_detector)
            future_flow = executor.submit(process_flow_threaded, resized_frame.copy(), flow_context)
>>>>>>> 9e85f966c9501cc0a2aa4d57d056c2a2f3733c03
            
            # Wait for all results (parallel execution happens here)
            t0 = time.time()
            yolo_result, yolo_confidence, yolo_ids, yolo_approach_detected = future_yolo.result()
<<<<<<< HEAD
            times_yolo.append(time.time() - t0)
            t0 = time.time()
            depth_color = future_depth.result()
            times_depth.append(time.time() - t0)
            t0 = time.time()
            flow_new, flow_ids, flow_uvs, flow_duvs = future_flow.result()
            times_flow.append(time.time() - t0)
=======
            flow_new, flow_ids, flow_uvs, flow_duvs = future_flow.result()
>>>>>>> 9e85f966c9501cc0a2aa4d57d056c2a2f3733c03
            
            frame_processing_time = time.time() - frame_start_time
            
            # Create combined display
            t0 = time.time()
            combined_frame = resized_frame.copy()

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
            times_combine.append(time.time() - t0)

            # Write frame if output is specified
            if writer:
                t0 = time.time()
                writer.write(display)
                times_write.append(time.time() - t0)
            else:
                times_write.append(0.0)
            
            if not args.no_display:
                cv2.imshow("DetectAndAvoid - YOLO | Optical Flow | ZipDepth", display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"frame_{frame_count:06d}.jpg", display)
                    print(f"Saved frame {frame_count}")
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
                _print_breakdown(
                    times_capture,
                    times_resize,
                    times_queue_wait,
                    times_yolo,
                    times_depth,
                    times_flow,
                    times_combine,
                    times_write,
                    zip_depth,
                    flow_context,
                    yolo_detector,
                )
        
            
            # Atualizar progresso
            # if frame_count % 30 == 0:
            #     elapsed_time = time.time() - total_processing_start_time
            #     avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            #     eta = ((elapsed_time / frame_count) * (total_frames - frame_count)) if frame_count > 0 else 0
            #     progress = (frame_count / total_frames) * 100
            #     print(f"Progresso: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
            #           f"FPS médio: {avg_fps:.2f} | ETA: {eta:.1f}s")
    except KeyboardInterrupt:
        stop_event.set()
        print("\nProcessing interrupted by user")
    
    except Exception as e:
        stop_event.set()
        print(f"Error during processing: {e}")
    
    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)
        executor.shutdown(wait=True)
        
        # Cleanup
        total_time = time.time() - total_processing_start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n--- Processing completed ---")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        _print_breakdown(
            times_capture,
            times_resize,
            times_queue_wait,
            times_yolo,
            times_depth,
            times_flow,
            times_combine,
            times_write,
            zip_depth,
            flow_context,
            yolo_detector,
        )
        
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
