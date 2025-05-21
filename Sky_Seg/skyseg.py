import os
import copy
import time
import cv2 as cv
import numpy as np
import onnxruntime
import argparse

def run_inference(onnx_session, model_input_target_size_hw, image_bgr):
    original_height, original_width = image_bgr.shape[:2]

    resized_image_for_model = cv.resize(
        image_bgr,
        dsize=(model_input_target_size_hw[1], model_input_target_size_hw[0]),
        interpolation=cv.INTER_AREA
    )

    rgb_image = cv.cvtColor(resized_image_for_model, cv.COLOR_BGR2RGB)
    normalized_image = np.array(rgb_image, dtype=np.float32)

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

    if max_val > min_val:
        normalized_mask = (output_mask_raw - min_val) / (max_val - min_val)
    else:
        normalized_mask = np.zeros_like(output_mask_raw)

    output_mask_uint8 = (normalized_mask * 255).astype('uint8')

    mask_resized_to_image_input_dims = cv.resize(
        output_mask_uint8,
        (original_width, original_height),
        interpolation=cv.INTER_NEAREST
    )
    return mask_resized_to_image_input_dims

# resulta video normal vs video segmentado lado a lado

def main(video_path):
    
    model_path = f"skyseg_fp16.onnx"  # "skyseg.onnx"
    video_input_path = video_path

    input_dir, filename = os.path.split(video_input_path)
    name, ext = os.path.splitext(filename)
    output_video_path = os.path.join(input_dir,  f"{name}_F16_segmentado{ext}")

    model_inference_input_size_hw = (320, 320)
    binary_threshold_value = 128

    if not os.path.exists(model_path):
        print(f"Error: ONNX model file not found at '{model_path}'.")
        return

    try:
        available_providers = onnxruntime.get_available_providers()
        preferred_providers_config = []

        if 'CUDAExecutionProvider' in available_providers:
            preferred_providers_config.append(
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  ######### quantidade da GB da GPU que deseja usar
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            )
        preferred_providers_config.append('CPUExecutionProvider')

        onnx_session = onnxruntime.InferenceSession(model_path, providers=preferred_providers_config)

        print(f"ONNX session using providers: {onnx_session.get_providers()}")
        if 'CUDAExecutionProvider' not in onnx_session.get_providers():
            print("Warning: CUDAExecutionProvider not used. Running on CPU.")

    except Exception as e:
        print(f"Error loading ONNX model: {e}. Trying CPU only.")
        try:
            onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print("ONNX session created with CPUExecutionProvider.")
        except Exception as e_cpu:
            print(f"Error loading ONNX model with CPU-only: {e_cpu}")
            return

    if args.video is None:
        print("Usando a câmera (cv2.VideoCapture(0))")
        cap = cv.VideoCapture(0)
    else:
        print(f"Usando o vídeo: {args.video}")
        cap = cv.VideoCapture(video_input_path)

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
    while(processing_frame_template.shape[0] >= 640 and processing_frame_template.shape[1] >= 640):
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

    frame_count = 0
    start_process_time = time.time()

    first_frame = True

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        original_view_frame = cv.resize(
            frame,
            (single_view_w, single_view_h),
            interpolation=cv.INTER_AREA
        )


        if(first_frame or frame_count % 30 == 0):
          first_frame = False

          segmentation_mask_gray = run_inference(
              onnx_session,
              model_inference_input_size_hw,
              original_view_frame
          )

        _, binary_mask = cv.threshold(
            segmentation_mask_gray,
            binary_threshold_value,
            255,
            cv.THRESH_BINARY
        )

        ###

        colored_image = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)

        # Pintar as áreas
        colored_image[binary_mask == 255] = (255, 105, 180)  # Rosa para área navegável
        colored_image[binary_mask == 0] = (144, 238, 144)    # Verde claro para área de perigo

        # Encontrar contornos para área branca (255)
        contours, _ = cv.findContours((binary_mask == 255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv.contourArea(cnt) > 100:  # Ignorar pequenos ruídos
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv.putText(colored_image, "Sky Area: Navigable", (cX-100, cY),
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Encontrar contornos para área preta (0)
        contours, _ = cv.findContours((binary_mask == 0).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv.contourArea(cnt) > 100:  # Ignorar pequenos ruídos
                M = cv.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv.putText(colored_image, "Danger: No-Navigation", (cX-100, cY),
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        ###

        #segmented_view_frame_bgr = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)

        combined_frame = np.hstack((original_view_frame, colored_image))

        out_video.write(combined_frame)

        frame_count += 1

    end_process_time = time.time()
    elapsed_time = end_process_time - start_process_time
    avg_fps_val = frame_count / elapsed_time if elapsed_time > 0 else float('inf')
    print(f"Avg FPS: {avg_fps_val:.2f}.")

    if frame_count > 0:
      print(f"Finished processing {frame_count} frames. Total time: {end_process_time - start_process_time:.2f}s")
    else:
      print("No frames were processed from the video.")

    print(f"Output video saved to: {output_video_path}")

    cap.release()
    out_video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Processador de vídeo ou câmera")
    parser.add_argument("video", nargs='?', default=None, help="Caminho para o vídeo ou use a câmera se não especificado")
    args = parser.parse_args()

    main(video_path=f"{args.video}")