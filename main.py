#!/usr/bin/env python3
"""
DeA2 - Optimized Detect-and-Avoid pipeline (Python + CUDA-friendly).

Key design goals:
- Keep latency low by always processing the newest frame.
- Avoid RTSP backlog with a continuous capture thread.
- Decouple module execution (YOLO / SkySeg / OpticalFlow) with async workers.
- Keep output video in real-time duration (no speed-up when processing is slower).
"""

import argparse
import os
import sys
import time
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


# ------------------------------- Paths ---------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEA1_DIR = REPO_ROOT / "DetectAndAvoid"

if str(DEA1_DIR) not in sys.path:
    sys.path.insert(0, str(DEA1_DIR))

from Yolo.Yolo11.modules.yolo_module import YOLODetector  # noqa: E402
from Yolo.Yolo11.modules.sky_seg_module import SkySegmentation  # noqa: E402


DEFAULT_YOLO_MODEL = DEA1_DIR / "Yolo" / "Yolo11" / "Weights" / "best_yolo_11_JUNHO_nano_drones_DGX.engine"
DEFAULT_SKY_MODEL = DEA1_DIR / "Sky_Seg" / "skyseg_fp16.onnx"
TRACKER_CONFIG = "bytetrack.yaml"


# ---------------------------- Performance setup ------------------------------
def configure_runtime() -> None:
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(1, min(os.cpu_count() or 1, 8)))
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="DeA2 optimized pipeline")
    parser.add_argument("--video-file", default=None, help="Input video file path (overrides RTSP when provided)")
    parser.add_argument("--video-ip", default="192.168.144.25", help="RTSP stream IP")
    parser.add_argument("--video-port", type=int, default=1945, help="RTSP stream port")
    parser.add_argument("--video-path", default="/", help="RTSP path (example: /live)")

    parser.add_argument("--resize-height", type=int, default=360, help="Processing frame height")
    parser.add_argument("--clusters", type=int, default=3, help="Optical flow clusters")
    parser.add_argument("--confidence", type=float, default=0.6, help="YOLO confidence threshold")

    parser.add_argument("--yolo-model-path", default=str(DEFAULT_YOLO_MODEL), help="YOLO .engine/.pt path")
    parser.add_argument("--sky-model-path", default=str(DEFAULT_SKY_MODEL), help="SkySeg ONNX path")
    parser.add_argument("--disable-cuda", action="store_true", help="Force CPU-only execution for DeA2 modules")
    parser.add_argument("--disable-sky", action="store_true", help="Disable Sky segmentation module")
    parser.add_argument("--disable-flow", action="store_true", help="Disable optical flow module")
    parser.add_argument("--flow-gpu", action="store_true", help="Use OpticalFlow GPU module")

    parser.add_argument("--yolo-update-interval", type=int, default=2, help="Run YOLO every N input frames")
    parser.add_argument("--sky-update-interval", type=int, default=3, help="Run SkySeg every N input frames")
    parser.add_argument("--flow-update-interval", type=int, default=1, help="Run OpticalFlow every N input frames")

    parser.add_argument("--output", help="Output video path")
    parser.add_argument("--output-fps", type=float, default=30.0, help="Output video FPS")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--stats-interval", type=float, default=2.0, help="Seconds between stats prints")
    parser.add_argument("--read-timeout", type=float, default=2.0, help="Seconds to wait for a new frame")

    return parser.parse_args()


# --------------------------- Capture / RTSP helpers --------------------------
def build_rtsp_url(ip: str, port: int, path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"rtsp://{ip}:{port}{normalized_path}"


def probe_capture_properties(cap: cv2.VideoCapture, tries: int = 40) -> Tuple[float, int, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width > 0 and height > 0:
        return fps, width, height

    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                return fps, w, h
        time.sleep(0.01)
    return fps, 0, 0


def open_capture(url: str) -> Tuple[cv2.VideoCapture, str, float, int, int]:
    urls = [url, url.rstrip("/")]
    urls = list(dict.fromkeys(urls))
    errors = []

    for u in urls:
        for proto in ("tcp", "udp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{proto}|stimeout;5000000|max_delay;500000"
            )
            cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if cap.isOpened():
                fps, width, height = probe_capture_properties(cap)
                if width > 0 and height > 0:
                    return cap, f"FFmpeg/{proto}", fps, width, height
                cap.release()
                errors.append(f"{u} via FFmpeg/{proto}: opened but no valid frame size")
            else:
                errors.append(f"{u} via FFmpeg/{proto}: open failed")

    msg = " | ".join(errors[-4:]) if errors else "no attempt details"
    raise RuntimeError(f"Could not open RTSP stream. Last attempts: {msg}")


def open_file_capture(video_file: str) -> Tuple[cv2.VideoCapture, str, float, int, int]:
    video_path = Path(video_file).expanduser().resolve()
    if not video_path.exists():
        raise RuntimeError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps, width, height = probe_capture_properties(cap)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Video opened but no valid frame size: {video_path}")

    return cap, f"FILE/{video_path.name}", fps, width, height


def resolve_fps(raw_fps: float, fallback: float = 30.0) -> float:
    if 1.0 <= raw_fps <= 120.0:
        return raw_fps
    return fallback


# ----------------------------- Async primitives ------------------------------
class LatestFrameReader:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.latest_frame = None
        self.latest_id = 0
        self.latest_ts = 0.0
        self.total_read = 0

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.002)
                continue
            now = time.perf_counter()
            with self.lock:
                self.latest_frame = frame
                self.latest_id += 1
                self.latest_ts = now
                self.total_read += 1
                self.new_frame_event.set()

    def get_latest(self, last_id: int, timeout: float):
        deadline = time.perf_counter() + timeout
        while not self.stop_event.is_set():
            with self.lock:
                if self.latest_frame is not None and self.latest_id != last_id:
                    return True, self.latest_frame.copy(), self.latest_id, self.latest_ts
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return False, None, last_id, 0.0
            self.new_frame_event.wait(min(0.01, remaining))
            self.new_frame_event.clear()
        return False, None, last_id, 0.0

    def stop(self):
        self.stop_event.set()
        self.new_frame_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class ModuleWorker:
    def __init__(self, name: str, process_fn: Callable):
        self.name = name
        self.process_fn = process_fn

        self.lock = threading.Lock()
        self.new_task_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.pending_frame = None
        self.pending_frame_id = 0
        self.has_pending = False

        self.last_output = None
        self.last_output_frame_id = 0
        self.last_proc_ms = 0.0
        self.last_error = None
        self.total_processed = 0

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame: np.ndarray, frame_id: int):
        with self.lock:
            self.pending_frame = frame
            self.pending_frame_id = frame_id
            self.has_pending = True
            self.new_task_event.set()

    def _run(self):
        while not self.stop_event.is_set():
            self.new_task_event.wait(0.05)
            self.new_task_event.clear()
            if self.stop_event.is_set():
                break

            with self.lock:
                if not self.has_pending:
                    continue
                frame = self.pending_frame
                frame_id = self.pending_frame_id
                self.has_pending = False

            t0 = time.perf_counter()
            try:
                out = self.process_fn(frame)
                err = None
            except Exception as exc:
                out = None
                err = str(exc)
            proc_ms = (time.perf_counter() - t0) * 1000.0

            with self.lock:
                if out is not None:
                    self.last_output = out
                    self.last_output_frame_id = frame_id
                    self.total_processed += 1
                self.last_proc_ms = proc_ms
                self.last_error = err

    def get_latest_output(self):
        with self.lock:
            return self.last_output, self.last_output_frame_id, self.last_proc_ms, self.last_error, self.total_processed

    def stop(self):
        self.stop_event.set()
        self.new_task_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class YOLODetectorCPU:
    """CPU-only fallback for YOLO when --disable-cuda is enabled."""
    def __init__(self, model_path: str, tracker_config: str, confidence_threshold: float):
        from ultralytics import YOLO

        selected_model = self._resolve_model_path(model_path)
        self.model = YOLO(selected_model)
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        print(f"✓ YOLO CPU model loaded: {selected_model}")

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"YOLO model not found: {path}")

        if path.suffix.lower() == ".engine":
            pt_candidate = path.with_suffix(".pt")
            if pt_candidate.exists():
                print(f"Info: --disable-cuda active, switching YOLO model to {pt_candidate.name}")
                return str(pt_candidate)
            raise RuntimeError(
                f"--disable-cuda requires a .pt model for YOLO. "
                f"Given: {path}. Provide --yolo-model-path <model.pt>."
            )

        return str(path)

    def process_frame(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
            conf=self.confidence_threshold,
            device="cpu",
        )
        if results and len(results) > 0:
            return results[0].plot(), False
        return frame, False


class SkySegmentationCPU:
    """CPU-only ONNX Runtime wrapper compatible with SkySegmentation interface."""
    def __init__(
        self,
        model_path: str,
        input_size=(320, 320),
        update_interval=1,
        sample_area_size=30,
        sky_upper_threshold=0.75,
        sky_lower_threshold=0.25,
        binary_threshold=128,
        use_tensorrt=False,
    ):
        import onnxruntime

        model = Path(model_path).expanduser().resolve()
        if not model.exists():
            raise RuntimeError(f"SkySeg model not found: {model}")

        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(
            str(model),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        print("✓ SkySeg running on CPUExecutionProvider")

        self.input_size = input_size
        self.update_interval = max(1, int(update_interval))
        self.binary_threshold = binary_threshold
        self.sample_area_size = sample_area_size
        self.sky_upper_threshold = sky_upper_threshold
        self.sky_lower_threshold = sky_lower_threshold

        self.frame_count = 0
        self.last_mask = None
        self.last_flight_status = "DESCONHECIDO"
        self.last_sky_ratio = 0.0
        self.last_roi_coords = (0, 0, 0, 0)
        self.last_status_color = (128, 128, 128)

    def _run_inference(self, image_bgr):
        original_h, original_w = image_bgr.shape[:2]
        resized = cv2.resize(image_bgr, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb / 255.0 - mean) / std
        input_tensor = normalized.transpose(2, 0, 1).reshape(1, 3, self.input_size[0], self.input_size[1]).astype(np.float32)

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        out = np.array(self.session.run([output_name], {input_name: input_tensor})).squeeze()

        min_val, max_val = float(np.min(out)), float(np.max(out))
        if max_val > min_val:
            out = (out - min_val) / (max_val - min_val)
        else:
            out = np.zeros_like(out)

        mask = (out * 255).astype("uint8")
        return cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    def _analyze(self, binary_mask):
        h, w = binary_mask.shape[:2]
        cy, cx = h // 2, w // 2
        half = self.sample_area_size // 2
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        roi = binary_mask[y1:y2, x1:x2]
        sky_ratio = float(np.mean(roi) / 255.0) if roi.size else 0.0

        if sky_ratio > self.sky_upper_threshold:
            status, color = "SUBINDO", (0, 255, 255)
        elif sky_ratio < self.sky_lower_threshold:
            status, color = "DESCENDO", (255, 0, 0)
        else:
            status, color = "NIVELADO", (0, 255, 0)
        return status, sky_ratio, (x1, y1, x2, y2), color

    def process_frame(self, frame):
        self.frame_count += 1
        should_update = (self.frame_count - 1) % self.update_interval == 0

        if should_update or self.last_mask is None:
            mask_gray = self._run_inference(frame)
            _, binary_mask = cv2.threshold(mask_gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
            self.last_flight_status, self.last_sky_ratio, self.last_roi_coords, self.last_status_color = self._analyze(binary_mask)
            self.last_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        display = self.last_mask.copy()
        x1, y1, x2, y2 = self.last_roi_coords
        cv2.rectangle(display, (x1, y1), (x2, y2), self.last_status_color, 2)
        cv2.putText(
            display,
            f"VOO: {self.last_flight_status} | CEU: {self.last_sky_ratio:.1%}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.last_status_color,
            2,
            cv2.LINE_AA,
        )
        return display, self.last_flight_status, self.last_sky_ratio


# ------------------------------ Frame helpers -------------------------------
def ensure_3ch(frame: Optional[np.ndarray], fallback: np.ndarray) -> np.ndarray:
    if frame is None:
        return fallback
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def put_shadow_text(frame, text, org, scale=0.6, fg=(255, 255, 255), bg=(0, 0, 0), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thick + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def main() -> int:
    configure_runtime()
    args = parse_args()

    if args.no_display and not args.output:
        print("Warning: --no-display sem --output roda apenas como benchmark.")

    print("=== DeA2 Optimized Pipeline ===")
    if args.video_file:
        print(f"Input file: {args.video_file}")
    else:
        rtsp_url = build_rtsp_url(args.video_ip, args.video_port, args.video_path)
        print(f"RTSP: {rtsp_url}")

    cap = None
    writer = None
    frame_reader = None
    yolo_worker = None
    sky_worker = None
    flow_worker = None
    flow_context = None
    flow_module = None

    try:
        if args.video_file:
            cap, backend_used, raw_fps, src_w, src_h = open_file_capture(args.video_file)
        else:
            cap, backend_used, raw_fps, src_w, src_h = open_capture(rtsp_url)
        print(f"Capture backend: {backend_used}")
        print(f"Source size: {src_w}x{src_h} | Reported FPS: {raw_fps}")

        if src_h <= 0:
            raise RuntimeError("Source height is zero after capture probe.")

        proc_h = int(args.resize_height)
        proc_w = int(src_w * (proc_h / float(src_h)))
        proc_size = (proc_w, proc_h)

        if proc_w <= 0 or proc_h <= 0:
            raise RuntimeError(f"Invalid processing size: {proc_size}")

        algo_fps = resolve_fps(raw_fps, fallback=30.0)
        output_fps = max(1.0, float(args.output_fps))

        print(f"Processing size: {proc_w}x{proc_h}")
        print(f"Algorithm FPS reference: {algo_fps}")
        print(f"Output FPS: {output_fps}")
        print(f"CUDA disabled: {args.disable_cuda}")

        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, output_fps, (proc_w * 3, proc_h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter: {args.output}")

        print("\n--- Initializing modules ---")
        if args.disable_cuda:
            yolo_detector = YOLODetectorCPU(
                model_path=args.yolo_model_path,
                tracker_config=TRACKER_CONFIG,
                confidence_threshold=args.confidence,
            )
        else:
            yolo_detector = YOLODetector(
                model_path=args.yolo_model_path,
                tracker_config=TRACKER_CONFIG,
                confidence_threshold=args.confidence,
                trail_length=50,
                approach_threshold=1.1,
                alert_duration=1.5,
                alert_message="# ALERTA: APROXIMACAO DETECTADA",
                alert_text_color=(0, 0, 255),
                alert_box_color=(0, 0, 0),
                alert_font_scale=1,
                alert_thickness=2,
            )

        sky_segmentation = None
        if not args.disable_sky:
            if args.disable_cuda:
                sky_segmentation = SkySegmentationCPU(
                    model_path=args.sky_model_path,
                    input_size=(320, 320),
                    update_interval=1,
                    sample_area_size=30,
                    sky_upper_threshold=0.75,
                    sky_lower_threshold=0.25,
                    binary_threshold=128,
                    use_tensorrt=False,
                )
            else:
                sky_segmentation = SkySegmentation(
                    model_path=args.sky_model_path,
                    input_size=(320, 320),
                    update_interval=1,
                    sample_area_size=30,
                    sky_upper_threshold=0.75,
                    sky_lower_threshold=0.25,
                    binary_threshold=128,
                    use_tensorrt=True,
                )

        if not args.disable_flow:
            use_flow_gpu = args.flow_gpu and not args.disable_cuda
            if args.flow_gpu and args.disable_cuda:
                print("Info: --disable-cuda active, forcing OpticalFlow CPU backend.")

            if use_flow_gpu:
                from OpticalFlow import opticalflow_gpu as flow_module  # noqa: WPS433
            else:
                from OpticalFlow import opticalflow as flow_module  # noqa: WPS433
            flow_context = flow_module.setup(
                clusters=args.clusters,
                fps=algo_fps,
                processing_size=proc_size,
            )

        print("Modules initialized.")

        yolo_worker = ModuleWorker("yolo", lambda f: yolo_detector.process_frame(f)[0])
        yolo_worker.start()

        if sky_segmentation is not None:
            sky_worker = ModuleWorker("sky", lambda f: sky_segmentation.process_frame(f)[0])
            sky_worker.start()

        if flow_context is not None and flow_module is not None:
            flow_worker = ModuleWorker("flow", lambda f: flow_module.process_frame(f, flow_context))
            flow_worker.start()

        frame_reader = LatestFrameReader(cap)
        frame_reader.start()

        print("\n--- Running ---")
        print("Press 'q' or ESC to exit.")

        frame_count = 0
        last_frame_id = 0
        start_ts = time.perf_counter()
        last_stats_ts = start_ts
        summary_printed = False

        writer_next_ts = time.perf_counter()

        while True:
            ok, frame, frame_id, frame_ts = frame_reader.get_latest(last_frame_id, timeout=args.read_timeout)
            if not ok:
                print("No new frames received in time. Stopping processing loop.")
                total_time = time.perf_counter() - start_ts
                avg_fps = frame_count / total_time if total_time > 0 else 0.0
                print("--- Processing completed ---")
                print(f"Total frames processed: {frame_count}")
                print(f"Total time: {total_time:.2f}s")
                print(f"Average FPS: {avg_fps:.2f}")
                summary_printed = True
                break

            last_frame_id = frame_id
            frame_count += 1
            resized = cv2.resize(frame, proc_size, interpolation=cv2.INTER_AREA)

            if frame_count % max(1, args.yolo_update_interval) == 0:
                yolo_worker.submit(resized, frame_id)
            if sky_worker is not None and frame_count % max(1, args.sky_update_interval) == 0:
                sky_worker.submit(resized, frame_id)
            if flow_worker is not None and frame_count % max(1, args.flow_update_interval) == 0:
                flow_worker.submit(resized, frame_id)

            yolo_frame, yolo_id, yolo_ms, yolo_err, yolo_total = yolo_worker.get_latest_output()
            sky_frame = None
            sky_id = 0
            sky_ms = 0.0
            sky_err = None
            sky_total = 0
            flow_frame = None
            flow_id = 0
            flow_ms = 0.0
            flow_err = None
            flow_total = 0

            if sky_worker is not None:
                sky_frame, sky_id, sky_ms, sky_err, sky_total = sky_worker.get_latest_output()
            if flow_worker is not None:
                flow_frame, flow_id, flow_ms, flow_err, flow_total = flow_worker.get_latest_output()

            yolo_view = ensure_3ch(yolo_frame, resized)
            sky_view = ensure_3ch(sky_frame, resized) if sky_worker is not None else resized
            flow_view = ensure_3ch(flow_frame, resized) if flow_worker is not None else resized

            combined = np.hstack([yolo_view, sky_view, flow_view])

            now = time.perf_counter()
            elapsed = max(1e-6, now - start_ts)
            loop_fps = frame_count / elapsed

            cap_fps = frame_reader.total_read / elapsed
            latency_ms = max(0.0, (now - frame_ts) * 1000.0)
            put_shadow_text(
                combined,
                f"in:{cap_fps:.1f}fps loop:{loop_fps:.1f} latency:{latency_ms:.0f}ms frame:{frame_count}",
                (10, 26),
                scale=0.6,
            )
            put_shadow_text(
                combined,
                f"YOLO {yolo_ms:.1f}ms lag:{max(0, frame_id - yolo_id)} proc:{yolo_total}",
                (10, 52),
                scale=0.55,
            )
            put_shadow_text(
                combined,
                f"SKY  {sky_ms:.1f}ms lag:{max(0, frame_id - sky_id)} proc:{sky_total}",
                (10, 76),
                scale=0.55,
            )
            put_shadow_text(
                combined,
                f"FLOW {flow_ms:.1f}ms lag:{max(0, frame_id - flow_id)} proc:{flow_total}",
                (10, 100),
                scale=0.55,
            )

            if yolo_err:
                put_shadow_text(combined, f"YOLO err: {yolo_err[:90]}", (10, proc_h - 54), scale=0.5, fg=(0, 0, 255))
            if sky_err:
                put_shadow_text(combined, f"SKY err: {sky_err[:90]}", (10, proc_h - 32), scale=0.5, fg=(0, 0, 255))
            if flow_err:
                put_shadow_text(combined, f"FLOW err: {flow_err[:90]}", (10, proc_h - 10), scale=0.5, fg=(0, 0, 255))

            if writer is not None:
                if now - writer_next_ts > 1.0:
                    writer_next_ts = now
                while now >= writer_next_ts:
                    writer.write(combined)
                    writer_next_ts += 1.0 / output_fps

            if not args.no_display:
                cv2.imshow("DeA2 - YOLO | SkySeg | OpticalFlow", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break

            if now - last_stats_ts >= args.stats_interval:
                print(
                    f"[stats] in={cap_fps:.1f}fps loop={loop_fps:.1f} "
                    f"yolo={yolo_ms:.1f}ms sky={sky_ms:.1f}ms flow={flow_ms:.1f}ms "
                    f"lag(y/s/f)=({max(0, frame_id - yolo_id)}/{max(0, frame_id - sky_id)}/{max(0, frame_id - flow_id)})"
                )
                last_stats_ts = now

        if not summary_printed:
            total_time = time.perf_counter() - start_ts
            avg_fps = frame_count / total_time if total_time > 0 else 0.0
            print("--- Processing completed ---")
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average FPS: {avg_fps:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"Fatal error: {exc}")
        return 1
    finally:
        if frame_reader is not None:
            frame_reader.stop()
        if yolo_worker is not None:
            yolo_worker.stop()
        if sky_worker is not None:
            sky_worker.stop()
        if flow_worker is not None:
            flow_worker.stop()

        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        if flow_context is not None and flow_module is not None:
            try:
                flow_module.cleanup(flow_context)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
