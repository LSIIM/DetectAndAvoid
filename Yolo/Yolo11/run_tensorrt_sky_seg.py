import os
import time
import copy
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort


# ---------- Utils ----------
def ort_dtype_to_numpy(ort_type: str):
    m = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
    }
    return m.get(ort_type, np.float32)


# ---------- Inference ----------
def build_session(model_path: str) -> ort.InferenceSession:
    os.makedirs("trt_cache", exist_ok=True)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    avail = ort.get_available_providers()
    providers = []

    if "TensorrtExecutionProvider" in avail:
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

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    print("Providers:", sess.get_providers())
    return sess


def preprocess(image_bgr: np.ndarray, size_hw: tuple[int, int], dtype) -> np.ndarray:
    h, w = size_hw
    img = cv.resize(image_bgr, (w, h), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Normalização ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # CHW
    img = img[np.newaxis, ...]    # NCHW
    img = np.ascontiguousarray(img).astype(dtype, copy=False)
    return img


def postprocess(mask_like: np.ndarray, out_size_wh: tuple[int, int]) -> np.ndarray:
    m = np.squeeze(mask_like)

    if m.ndim == 3:  # (C,H,W)
        if m.shape[0] > 1:
            m = np.argmax(m, axis=0).astype(np.float32)
        else:
            m = m[0]
    elif m.ndim != 2:
        m = m.reshape(m.shape[-2], m.shape[-1])

    if np.issubdtype(m.dtype, np.integer):
        m = (m > 0).astype(np.uint8) * 255
    else:
        m = m.astype(np.float32)
        mn, mx = float(np.min(m)), float(np.max(m))
        if mx > mn:
            m = (m - mn) / (mx - mn)
        else:
            m = np.zeros_like(m, dtype=np.float32)
        m = (m * 255.0).astype(np.uint8)

    m = cv.resize(m, out_size_wh, interpolation=cv.INTER_NEAREST)
    return m


def run_inference(sess: ort.InferenceSession, size_hw: tuple[int, int], image_bgr: np.ndarray) -> np.ndarray:
    h0, w0 = image_bgr.shape[:2]

    in0 = sess.get_inputs()[0]
    in_name = in0.name
    in_dtype = ort_dtype_to_numpy(in0.type)  # <-- usa dtype exigido pelo modelo (FP16 no seu .onnx)
    out_name = sess.get_outputs()[0].name

    inp = preprocess(image_bgr, size_hw, in_dtype)
    out = sess.run([out_name], {in_name: inp})[0]

    mask = postprocess(out, (w0, h0))
    return mask


# ---------- App ----------
def main():
    model_path = r"Weights/skyseg_fp16.onnx"          # seu modelo FP16
    video_input_path = r"videos_test/fev_corte_3.mp4"
    model_inference_input_size_hw = (320, 320)
    binary_threshold_value = 128

    if not Path(model_path).exists():
        print(f"ERRO: modelo ONNX não encontrado: {model_path}")
        return
    if not Path(video_input_path).exists():
        print(f"ERRO: vídeo não encontrado: {video_input_path}")
        return

    try:
        sess = build_session(model_path)
    except Exception as e:
        print(f"ERRO criando sessão ORT/TRT: {e}")
        return

    cap = cv.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"ERRO: não abriu vídeo: {video_input_path}")
        return

    fps_src = cap.get(cv.CAP_PROP_FPS) or 30.0
    ret, first = cap.read()
    if not ret:
        print("ERRO: não leu primeiro frame.")
        cap.release()
        return

    # layout de saída
    tmpl = copy.deepcopy(first)
    while tmpl.shape[0] >= 640 and tmpl.shape[1] >= 640:
        tmpl = cv.pyrDown(tmpl)
    h1, w1 = tmpl.shape[:2]
    out_w, out_h = w1 * 2, h1

    # writer
    out_path = str(Path(video_input_path).with_name(Path(video_input_path).stem + "_TRT_Provider_segmentado.mp4"))
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(out_path, fourcc, fps_src, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERRO: não abriu writer: {out_path}")
        cap.release()
        return

    # loop
    frame_count = 0
    t0 = time.time()
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_small = cv.resize(frame, (w1, h1), interpolation=cv.INTER_AREA)
            mask_gray = run_inference(sess, model_inference_input_size_hw, frame_small)

            # binário
            _, mask_bin = cv.threshold(mask_gray, binary_threshold_value, 255, cv.THRESH_BINARY)
            mask_bgr = cv.cvtColor(mask_bin, cv.COLOR_GRAY2BGR)

            stacked = np.hstack((frame_small, mask_bgr))
            fps_avg = (frame_count + 1) / max(time.time() - t0, 1e-6)
            cv.putText(stacked, f"FPS: {fps_avg:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv.LINE_AA)

            cv.imshow("Processing (ORT + TensorRT)", stacked)
            if (cv.waitKey(1) & 0xFF) in (27, ord("q")):
                break

            writer.write(stacked)
            frame_count += 1
    finally:
        t1 = time.time()
        cap.release()
        writer.release()
        cv.destroyAllWindows()
        if frame_count:
            print(f"Frames: {frame_count}  Tempo: {t1 - t0:.2f}s  FPS médio: {frame_count / (t1 - t0):.2f}")
        print(f"Saída: {out_path}")


if __name__ == "__main__":
    main()
