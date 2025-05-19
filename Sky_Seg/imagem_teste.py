import os
import copy
import time
import argparse
import cv2 as cv
import numpy as np
import onnxruntime





def run_inference(onnx_session, input_size_hw, image_bgr):
    original_height, original_width = image_bgr.shape[:2]

    resized_image = cv.resize(
        image_bgr,
        dsize=(input_size_hw[1], input_size_hw[0]),
        interpolation=cv.INTER_AREA
    )

    rgb_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)

    normalized_image = np.array(rgb_image, dtype=np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    normalized_image = (normalized_image / 255.0 - mean) / std

    transposed_image = normalized_image.transpose(2, 0, 1)

    input_tensor = transposed_image.reshape(1, 3, input_size_hw[0], input_size_hw[1]).astype('float32')

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    onnx_result = onnx_session.run([output_name], {input_name: input_tensor})

    output_mask = np.array(onnx_result).squeeze()

    min_val = np.min(output_mask)
    max_val = np.max(output_mask)

    if max_val > min_val:
        output_mask = (output_mask - min_val) / (max_val - min_val)
    else:
        output_mask = np.zeros_like(output_mask)

    output_mask *= 255
    output_mask_uint8 = output_mask.astype('uint8')

    result_map_resized = cv.resize(
        output_mask_uint8,
        (original_width, original_height),
        interpolation=cv.INTER_NEAREST
    )

    return result_map_resized


def main():
    model_path = "skyseg.onnx"
    image_path = r"imgs_test\fotos_drone1.jpg"
    
    input_dir, filename = os.path.split(image_path)
    name, ext = os.path.splitext(filename)
    output_image_path = os.path.join(input_dir, f"{name}_segmentada{ext}")


    if not os.path.exists(model_path):
        print(f"Error: ONNX model file not found at '{model_path}'. Please upload it.")
        return
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at '{image_path}'. Please upload it.")
        return

    image_bgr = cv.imread(image_path)
    if image_bgr is None:
        print(f"Error loading image from: {image_path}")
        return

    processed_image_bgr = copy.deepcopy(image_bgr)
    while(processed_image_bgr.shape[0] >= 640 and processed_image_bgr.shape[1] >= 640):
        processed_image_bgr = cv.pyrDown(processed_image_bgr)

    try:
        onnx_session = onnxruntime.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    model_input_size_hw = (320, 320)

    start_time = time.time()
    result_map = run_inference(onnx_session, model_input_size_hw, processed_image_bgr)
    end_time = time.time()

    print(f"Inference time: {end_time - start_time:.4f} seconds")

    cv.imwrite(output_image_path, result_map)
    print(f"Result map saved to: {output_image_path}")

    if len(result_map.shape) == 2:
        result_map_display_bgr = cv.cvtColor(result_map, cv.COLOR_GRAY2BGR)
    else:
        result_map_display_bgr = result_map

    combined_display = np.hstack((processed_image_bgr, result_map_display_bgr))
    cv.imshow("Sky Segmentation", combined_display)
    cv.waitKey(0)

if __name__ == '__main__':
    main()
