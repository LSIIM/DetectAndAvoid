"""
MiDaS v2.1 small depth inference using TensorRT.

Does not use pycuda.autoinit so the CUDA context can be shared with YOLO
on worker threads.
"""

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

INPUT_SIZE = 256  # MiDaS small input (256x256)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        self.input_nbytes = int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize
        self.output_nbytes = int(np.prod(self.output_shape)) * np.dtype(np.float32).itemsize

        self.h_input = cuda.pagelocked_empty(int(np.prod(self.input_shape)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(int(np.prod(self.output_shape)), dtype=np.float32)

        self.d_input = cuda.mem_alloc(self.input_nbytes)
        self.d_output = cuda.mem_alloc(self.output_nbytes)

        self.stream = cuda.Stream()

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

    def infer(self, input_array):
        np.copyto(self.h_input, input_array.ravel())

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.h_output.reshape(self.output_shape)


def preprocess(frame_bgr, input_size=INPUT_SIZE):
    """Convert BGR frame to the tensor expected by MiDaS."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img, dtype=np.float32)


def postprocess(depth, original_shape, colormap=cv2.COLORMAP_INFERNO):
    """Colorize at 256x256, then upsample the visualization."""
    depth = np.squeeze(depth)

    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)

    depth_vis = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, colormap)
    out_w, out_h = original_shape[1], original_shape[0]
    if depth_color.shape[1] != out_w or depth_color.shape[0] != out_h:
        depth_color = cv2.resize(depth_color, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return depth_color


class MiDaSDepth:
    """MiDaS v2.1 small depth estimator (TensorRT)."""

    def __init__(self, model_path, input_size=INPUT_SIZE, warmup_iters=5):
        """
        Args:
            model_path: Path to TensorRT engine (.engine)
            input_size: Model input size (default 256)
            warmup_iters: Dummy inferences after load
        """
        self.input_size = input_size

        cuda.init()
        self.cfx = cuda.Device(0).retain_primary_context()
        self.cfx.push()
        try:
            self.engine = TRTEngine(model_path)
            print(
                f"✓ MiDaS TensorRT loaded | input {self.engine.input_shape} "
                f"| output {self.engine.output_shape}"
            )
            dummy = np.zeros((input_size, input_size, 3), dtype=np.uint8)
            inp = preprocess(dummy, input_size=self.input_size)
            for _ in range(warmup_iters):
                self.engine.infer(inp)
            print("✓ MiDaS warm-up complete")
        finally:
            self.cfx.pop()

    def process_frame(self, frame):
        """
        Run depth inference on a BGR frame.

        Returns:
            depth_color: BGR uint8 visualization, same HxW as input
        """
        self.cfx.push()
        try:
            inp = preprocess(frame, input_size=self.input_size)
            depth_output = self.engine.infer(inp)
            return postprocess(depth_output, frame.shape[:2])
        finally:
            self.cfx.pop()
