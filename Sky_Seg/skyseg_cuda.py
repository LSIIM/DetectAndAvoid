import os
import copy
import time
#Aqui eh o path para minhas dlls OpenCV Cuda + Cuda, se não tiver setado via path, tem q mudar o endereço.
#os.add_dll_directory(r"C:\LSIIM\sky_seg\opencv_dlls")
import cv2 as cv
import numpy as np
import onnxruntime
import argparse

def run_inference(onnx_session, model_input_target_size_hw, image_bgr):
    original_height, original_width = image_bgr.shape[:2]

    gpu_bgr = cv.cuda_GpuMat()
    gpu_bgr.upload(image_bgr)

    gpu_resized = cv.cuda.resize(
        gpu_bgr,
        (model_input_target_size_hw[1], model_input_target_size_hw[0]),
        interpolation=cv.INTER_LINEAR
    )

    gpu_rgb = cv.cuda.cvtColor(gpu_resized, cv.COLOR_BGR2RGB)
    resized_image_for_model = gpu_rgb.download()

    normalized_image = np.array(resized_image_for_model, dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_image = (normalized_image / 255.0 - mean) / std
    transposed_image = normalized_image.transpose(2, 0, 1)
    input_tensor = transposed_image.reshape(1, 3, model_input_target_size_hw[0], model_input_target_size_hw[1]).astype(np.float16)

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    onnx_result = onnx_session.run([output_name], {input_name: input_tensor})
    output_mask_raw = np.array(onnx_result).squeeze()

    min_val = np.min(output_mask_raw)
    max_val = np.max(output_mask_raw)
    normalized_mask = (output_mask_raw - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(output_mask_raw)
    output_mask_uint8 = (normalized_mask * 255).astype('uint8')

    mask_resized_to_image_input_dims = cv.resize(
        output_mask_uint8,
        (original_width, original_height),
        interpolation=cv.INTER_NEAREST
    )
    return mask_resized_to_image_input_dims

def main(video_path):
    model_path = f"skyseg_fp16.onnx"
    video_input_path = video_path
    input_dir, filename = os.path.split(video_input_path)
    name, ext = os.path.splitext(filename)
    output_video_path = os.path.join(input_dir, f"{name}_F16_segmentado{ext}")

    model_inference_input_size_hw = (320, 320)
    binary_threshold_value = 128

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 2
    options.enable_cpu_mem_arena = False
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = True

    if not os.path.exists(model_path):
        print(f"Error: ONNX model file not found at '{model_path}'.")
        return

    try:
        available_providers = onnxruntime.get_available_providers()
        preferred_providers_config = []

        #tirar comentário abaixo para rodar na Jetson e usar TensorRT

        """
        if 'TensorrtExecutionProvider' in available_providers:
            print("Iniciando TensorRT...")
            preferred_providers_config.append(('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 1 << 30,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': "./trt_cache",
                'trt_dla_enable': False,
            }))
        """
        if 'CUDAExecutionProvider' in available_providers:
            preferred_providers_config.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
        preferred_providers_config.append('CPUExecutionProvider')

        onnx_session = onnxruntime.InferenceSession(model_path, providers=preferred_providers_config, sess_options=options)
        print(f"ONNX session using providers: {onnx_session.get_providers()}")

    except Exception as e:
        print(f"Error loading ONNX model: {e}. Trying CPU only.")
        try:
            onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e_cpu:
            print(f"Error loading ONNX model with CPU-only: {e_cpu}")
            return

    cap = cv.VideoCapture(0 if args.video is None else video_input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_input_path}")
        return

    fps = cap.get(cv.CAP_PROP_FPS)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from video.")
        cap.release()
        return

    processing_frame_template = copy.deepcopy(first_frame)
    while processing_frame_template.shape[0] >= 640 and processing_frame_template.shape[1] >= 640:
        processing_frame_template = cv.pyrDown(processing_frame_template)

    single_view_h, single_view_w = processing_frame_template.shape[:2]
    output_video_w = single_view_w * 2
    output_video_h = single_view_h
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out_video = cv.VideoWriter(output_video_path, fourcc, fps, (output_video_w, output_video_h))
    if not out_video.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}.")
        cap.release()
        return

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_count, start_process_time = 0, time.time()
    first_frame_flag = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_view_frame = cv.resize(frame, (single_view_w, single_view_h), interpolation=cv.INTER_LINEAR)

        if first_frame_flag or frame_count % 30 == 0:
            first_frame_flag = False
            segmentation_mask_gray = run_inference(onnx_session, model_inference_input_size_hw, original_view_frame)

            gpu_mask = cv.cuda_GpuMat()
            gpu_mask.upload(segmentation_mask_gray)
            _, gpu_binary = cv.cuda.threshold(gpu_mask, binary_threshold_value, 255, cv.THRESH_BINARY)

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            morph_filter = cv.cuda.createMorphologyFilter(cv.MORPH_OPEN, gpu_binary.type(), kernel)
            gpu_cleaned = morph_filter.apply(gpu_binary)

            binary_mask = gpu_binary.download()
            cleaned_mask = gpu_cleaned.download()

        colored_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
        colored_image[binary_mask == 255] = (255, 105, 180)
        colored_image[binary_mask == 0] = (144, 238, 144)

        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(cleaned_mask, 8, cv.CV_32S)
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            if area > 100:
                cX, cY = int(centroids[i, 0]), int(centroids[i, 1])
                text = "Sky Area: Navigable" if binary_mask[cY, cX] == 255 else "Danger: No-Navigation"
                cv.putText(colored_image, text, (cX - 100, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        combined_frame = np.hstack((original_view_frame, colored_image))
        out_video.write(combined_frame)
        frame_count += 1

    elapsed_time = time.time() - start_process_time
    print(f"Avg FPS: {frame_count / elapsed_time:.2f}.")
    print(f"Finished processing {frame_count} frames. Total time: {elapsed_time:.2f}s")
    print(f"Output video saved to: {output_video_path}")

    cap.release()
    out_video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processador de vídeo ou câmera")
    parser.add_argument("video", nargs='?', default=None, help="Caminho para o vídeo ou use a câmera se não especificado")
    args = parser.parse_args()
    main(video_path=f"{args.video}")
