#!/usr/bin/env python3
"""
DetectAndAvoid - Main Integration Module

This module integrates YOLO detection, ZipDepth, and Optical Flow
processing into a unified video processing pipeline.

Usage:
    python main.py <video_path> [--clusters <num>] [--confidence <conf>]
"""

import argparse
from collections import deque
import json
import os
import queue
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.YOLO.yolo_module import YOLODetector, kalman_filter
from modules.depth.zip_depth_module import ZipDepth, extract_roi_depth
# from modules.Optical_Flow import opticalflow as optical_flow
from modules.Optical_Flow.opticalflow_roi import RoiPointTracker

YOLO_MODEL_PATH = r"weights/best_yolo26_drone_bird_aircraft_junho_2026.engine"
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
    parser.add_argument("--visual-depth", action="store_true", help="Colorize depth and write the side-by-side debug video")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame JSON and latency breakdowns")

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
class LatestFrame:
    """Guarda só o frame mais recente. A publicação substitui o anterior."""

    def __init__(self):
        self._cv = threading.Condition()
        self._seq = 0
        self._frame = None
        self._ts = None

    def publish(self, frame, frame_ts):
        with self._cv:
            self._seq += 1
            self._frame = frame
            self._ts = frame_ts
            self._cv.notify()

    def wait_newer(self, last_seq, timeout):
        with self._cv:
            if self._frame is None or self._seq == last_seq:
                self._cv.wait(timeout)
            if self._frame is None or self._seq == last_seq:
                return None
            return self._frame.copy(), self._ts, self._seq

    def wake(self):
        with self._cv:
            self._cv.notify_all()


class LatestDetection:
    """Última detecção publicada. A leitura não bloqueia."""

    def __init__(self):
        self._lock = threading.Lock()
        self._value = None
        self.count = 0

    def publish(self, boxes, confidences, classes, ids, approach, frame_ts):
        with self._lock:
            self._value = (boxes, confidences, classes, ids, approach, frame_ts)
            self.count += 1

    def read(self):
        with self._lock:
            return self._value

    def completed(self):
        with self._lock:
            return self.count


def _enqueue_latest(record_q, packet):
    """Enfileira sem bloquear. Se a fila enche, descarta o pacote mais antigo."""
    try:
        record_q.put_nowait(packet)
        return
    except queue.Full:
        pass
    try:
        record_q.get_nowait()
    except queue.Empty:
        pass
    try:
        record_q.put_nowait(packet)
    except queue.Full:
        pass


def _render_record_frame(frame_bgr, tracks, fps, info_text, depth_color):
    combined = frame_bgr.copy()
    for track in tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in track["box"]]
        color = (0, 255, 0) if track["status"] == "updated" else (0, 165, 255)
        cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            combined, str(track["track_id"]), (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
        dx = track.get("dx") or 0.0
        dy = track.get("dy") or 0.0
        cx = int(round(track["cx"]))
        cy = int(round(track["cy"]))
        cv2.circle(combined, (cx, cy), 5, (0, 0, 0), -1)
        cv2.arrowedLine(
            combined, (cx, cy),
            (int(round(cx + dx * fps)), int(round(cy + dy * fps))),
            (0, 0, 0), 2, tipLength=0.2,
        )
    cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    if depth_color is None:
        return combined
    return np.hstack([combined, depth_color])


def process_depth_threaded(frame, zip_depth):
    """Process ZipDepth in a separate thread"""
    try:
        return zip_depth.process_frame(frame)
    except Exception as e:
        print(f"Error in ZipDepth processing: {e}")
        return np.zeros_like(frame)

def _print_breakdown(
    times_capture,
    times_resize,
    times_queue_wait,
    times_depth,
    times_flow_roi,
    times_combine,
    times_write,
    zip_depth=None,
    flow_context=None,
    yolo_detector=None,
    main_fps=None,
    yolo_fps=None,
    detection_ages=None,
):
    def _line(label, xs):
        if not xs:
            return
        print(f"{label:22s} {np.mean(xs) * 1000:6.2f} ms/frame")

    print("\n--- Breakdown médio por etapa ---")
    _line("Capture:", times_capture)
    _line("Resize (CPU shared):", times_resize)
    _line("Queue wait:", times_queue_wait)
    _line("Depth (wait result):", times_depth)
    _line("Flow ROI:", times_flow_roi)
    _line("Combine:", times_combine)
    _line("Write:", times_write)
    if yolo_detector is not None:
        _line("YOLO preprocess:", getattr(yolo_detector, "_times_preprocess", None))
        _line("YOLO inference:", getattr(yolo_detector, "_times_inference", None))
        _line("YOLO postprocess:", getattr(yolo_detector, "_times_postprocess", None))
        _line("YOLO predict() total:", getattr(yolo_detector, "_times_predict_total", None))
        _line("YOLO post loop:", getattr(yolo_detector, "_times_post_loop", None))
    if zip_depth is not None:
        _line("Depth infer:", getattr(zip_depth, "_times_infer", None))
        _line("Depth postprocess:", getattr(zip_depth, "_times_postprocess", None))
        _line("Depth colorize:", getattr(zip_depth, "_times_colorize", None))
    if main_fps is not None:
        print(f"{'Main FPS:':22s} {main_fps:6.2f}")
    if yolo_fps is not None:
        print(f"{'YOLO FPS:':22s} {yolo_fps:6.2f}")
    if detection_ages:
        print(
            f"{'Det age ms:':22s} "
            f"mean {np.mean(detection_ages):6.1f}  "
            f"min {np.min(detection_ages):6.1f}  "
            f"max {np.max(detection_ages):6.1f}"
        )


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
    
    out_width = processing_width * 2 if args.visual_depth else processing_width
    writer = setup_video_writer(args.output, fps, out_width, processing_height)

    
    # Setup modules
    print("\n--- Setting up modules ---")
    
    try:        
        # Optical Flow setup
        print("Setting up Optical Flow...")
        flow_context = None
        print("Setting up YOLO detector...")
        tracker = kalman_filter()
        yolo_detector = YOLODetector(
            model_path=args.yolo_model_path,
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
            model_path=args.depth_model_path,
            visual=args.visual_depth,
        )
        
        print("All modules setup successfully!")
        
    except Exception as e:
        print(f"Error setting up modules: {e}")
        cap.release()
        if writer:
            writer.release()
        return 1
    
    # Depth na pool. YOLO e flow têm thread própria.
    executor = ThreadPoolExecutor(max_workers=1)
    
    # Main processing loop
    print("\n--- Starting video processing ---")
    print("Using capture + YOLO thread + flow thread + depth + record")
    record_q = queue.Queue(maxsize=8)

    def record_worker():
        log_file = open("tracks.jsonl", "w", encoding="utf-8")
        try:
            while True:
                packet = record_q.get()
                if packet is None:
                    break
                log_file.write(json.dumps(packet["log"]) + "\n")
                log_file.flush()
                frame = packet.get("frame")
                if writer is None or frame is None:
                    continue
                depth_color = None
                if args.visual_depth and packet.get("depth") is not None:
                    depth_color = zip_depth.colorize(packet["depth"], frame.shape[:2])
                display = _render_record_frame(
                    frame, packet["tracks"], fps, packet["info_text"], depth_color,
                )
                if packet["frame_count"] % 30 == 0:
                    cv2.imwrite(
                        f"audit_tracks/frame_{packet['frame_count']:06d}.jpg",
                        display if depth_color is None else display[:, : frame.shape[1]],
                    )
                writer.write(display)
        finally:
            log_file.close()
            if writer:
                writer.release()

    record_thread = threading.Thread(target=record_worker, daemon=True)
    record_thread.start()
    frame_count = 0
    total_processing_start_time = time.time()
    times_capture = []
    times_resize = []
    times_queue_wait = []
    times_depth = []
    times_flow_roi = []
    times_combine = []
    times_write = []

    SENTINEL = None
    frame_q = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    latest_frame = LatestFrame()
    latest_det = LatestDetection()
    detection_ages = []
    last_det_ts = None
    prev_gray = None
    prev_depth = {}
    depth_window = 8
    flow_jobs = queue.Queue()
    os.makedirs("audit_tracks", exist_ok=True)

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
                frame_ts = t0
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
                if not _put_frame((frame_id, resized_frame, frame_ts)):
                    break
        finally:
            _put_frame(SENTINEL)

    def yolo_worker():
        last_seq = 0
        while not stop_event.is_set():
            item = latest_frame.wait_newer(last_seq, 0.2)
            if item is None:
                continue
            frame, det_frame_ts, seq = item
            last_seq = seq
            try:
                boxes, confidences, classes, ids, approach = yolo_detector.process_frame(frame)
            except Exception as e:
                print(f"Error in YOLO processing: {e}")
                continue
            latest_det.publish(boxes, confidences, classes, ids, approach, det_frame_ts)

    def flow_worker():
        roi_flow = RoiPointTracker()
        while True:
            job = flow_jobs.get()
            if job is None:
                break
            items, prev, curr, reply = job
            try:
                results, elapsed = roi_flow.step(items, prev, curr)
            except Exception as e:
                print(f"Error in Optical Flow processing: {e}")
                results, elapsed = [], 0.0
            reply.put((results, elapsed))

    capture_thread = threading.Thread(target=capture_worker, name="capture", daemon=True)
    capture_thread.start()
    yolo_thread = threading.Thread(target=yolo_worker, name="yolo", daemon=True)
    yolo_thread.start()
    flow_thread = threading.Thread(target=flow_worker, name="flow", daemon=True)
    flow_thread.start()
    
    try:
        while True:
            t0 = time.time()
            item = frame_q.get()
            times_queue_wait.append(time.time() - t0)
            if item is SENTINEL:
                break
            frame_count, resized_frame, frame_ts = item
            frame_start_time = time.time()
            latest_frame.publish(resized_frame, frame_ts)

            snapshot = latest_det.read()
            yolo_result = None
            yolo_confidence = None
            yolo_ids = []
            det_ts = None
            if snapshot is not None:
                yolo_result, yolo_confidence, yolo_classes, yolo_ids, yolo_approach_detected, det_ts = snapshot
                age_ms = (frame_ts - det_ts) * 1000.0
                detection_ages.append(age_ms)

            measurement_boxes = None
            if det_ts is not None and det_ts != last_det_ts:
                last_det_ts = det_ts
                measurement_boxes = yolo_result if yolo_result is not None else np.empty((0, 4), dtype=int)
                meas_classes = yolo_classes
                meas_conf = yolo_confidence
            else:
                meas_classes = None
                meas_conf = None
            tracker.step(
                frame_ts, measurement_boxes, yolo_detector._assign_track_ids,
                meas_classes, meas_conf,
            )
            
            curr_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            flow_items = [
                (track_id, track["box"]) for track_id, track in tracker.tracks.items()
            ]
            flow_reply = queue.Queue(maxsize=1)
            flow_jobs.put((flow_items, prev_gray, curr_gray, flow_reply))

            # Shared buffer: workers do not write the color frame
            future_depth = executor.submit(process_depth_threaded, resized_frame, zip_depth)
            
            t0 = time.time()
            depth_output = future_depth.result()
            times_depth.append(time.time() - t0)
            t0 = time.time()
            active_ids = set(tracker.tracks)
            depth_stats = {}
            for track_id, track in tracker.tracks.items():
                if depth_output is None:
                    continue
                s_t = extract_roi_depth(
                    depth_output,
                    track["box"],
                    zip_depth.input_size,
                    processing_width,
                    processing_height,
                )
                if s_t is None:
                    continue
                history = prev_depth.setdefault(track_id, deque(maxlen=depth_window))
                history.append((s_t, frame_ts))
                level = float(np.median([sample for sample, _ts in history]))
                if len(history) < depth_window:
                    depth_stats[track_id] = (level, None)
                else:
                    half = depth_window // 2
                    older = [sample for sample, _ts in list(history)[:half]]
                    newer = [sample for sample, _ts in list(history)[half:]]
                    delta_s = float(np.median(newer) - np.median(older))
                    depth_stats[track_id] = (level, delta_s)
            for track_id in list(prev_depth):
                if track_id not in active_ids:
                    del prev_depth[track_id]
            zip_depth._times_postprocess.append(time.time() - t0)
            flow_results, flow_dt = flow_reply.get()
            times_flow_roi.append(flow_dt)
            prev_gray = curr_gray

            frame_processing_time = time.time() - frame_start_time
            flow_by_id = {
                track_id: (dx, dy) for track_id, dx, dy, _magnitude, _angle in flow_results
            }
            log_tracks = []
            draw_tracks = []
            for track_id, track in tracker.tracks.items():
                kalman = tracker.kalman_filters[track_id]
                state = kalman.statePost if track["status"] == "updated" else kalman.statePre
                cx = float(state[0, 0])
                cy = float(state[1, 0])
                vx = float(state[2, 0])
                vy = float(state[3, 0])
                level, delta_s = depth_stats.get(track_id, (None, None))
                dx, dy = flow_by_id.get(track_id, (0.0, 0.0))
                log_tracks.append({
                    "track_id": int(track_id),
                    "classe": track.get("classe"),
                    "confianca": track.get("confianca"),
                    "cx": cx,
                    "cy": cy,
                    "vx": vx,
                    "vy": vy,
                    "S": level,
                    "deltaS": delta_s,
                    "timestamp": frame_ts,
                })
                draw_tracks.append({
                    "track_id": int(track_id),
                    "box": [float(v) for v in track["box"]],
                    "status": track["status"],
                    "cx": cx,
                    "cy": cy,
                    "dx": dx,
                    "dy": dy,
                })
            info_text = f"Frame: {frame_count} | YOLO | ZipDepth | Optical Flow | {frame_processing_time*1000:.1f}ms"
            packet = {
                "log": {"frame": frame_count, "timestamp": frame_ts, "tracks": log_tracks},
                "frame_count": frame_count,
                "info_text": info_text,
                "tracks": draw_tracks,
                "frame": resized_frame.copy() if writer or not args.no_display else None,
                "depth": depth_output if args.visual_depth else None,
            }
            if args.verbose:
                print(json.dumps(packet["log"]))
            t0 = time.time()
            _enqueue_latest(record_q, packet)
            times_write.append(time.time() - t0)
            times_combine.append(0.0)

            if not args.no_display and packet["frame"] is not None:
                depth_color = None
                if args.visual_depth and depth_output is not None:
                    depth_color = zip_depth.colorize(depth_output, packet["frame"].shape[:2])
                display = _render_record_frame(
                    packet["frame"], draw_tracks, fps, info_text, depth_color,
                )
                cv2.imshow("DetectAndAvoid - YOLO | Optical Flow | ZipDepth", display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"frame_{frame_count:06d}.jpg", display)
                    print(f"Saved frame {frame_count}")
            
            # Print progress every 100 frames
            if args.verbose and frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
                elapsed = time.time() - total_processing_start_time
                _print_breakdown(
                    times_capture,
                    times_resize,
                    times_queue_wait,
                    times_depth,
                    times_flow_roi,
                    times_combine,
                    times_write,
                    zip_depth,
                    flow_context,
                    yolo_detector,
                    main_fps=(frame_count / elapsed) if elapsed > 0 else 0.0,
                    yolo_fps=(latest_det.completed() / elapsed) if elapsed > 0 else 0.0,
                    detection_ages=detection_ages,
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
        latest_frame.wake()
        flow_jobs.put(None)
        record_q.put(None)
        capture_thread.join(timeout=2.0)
        yolo_thread.join(timeout=2.0)
        flow_thread.join(timeout=2.0)
        record_thread.join(timeout=5.0)
        executor.shutdown(wait=True)
        
        # Cleanup
        total_time = time.time() - total_processing_start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        yolo_fps = latest_det.completed() / total_time if total_time > 0 else 0
        
        if args.verbose:
            print(f"\n--- Processing completed ---")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average FPS: {avg_fps:.2f}")
            _print_breakdown(
                times_capture,
                times_resize,
                times_queue_wait,
                times_depth,
                times_flow_roi,
                times_combine,
                times_write,
                zip_depth,
                flow_context,
                yolo_detector,
                main_fps=avg_fps,
                yolo_fps=yolo_fps,
                detection_ages=detection_ages,
            )
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup modules
        # optical_flow.cleanup(flow_context)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())