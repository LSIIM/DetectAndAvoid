# Sky Segmentation Project

## Overview
- This project implements **sky segmentation** in real time using **ONNX Runtime** and **OpenCV with CUDA** acceleration.  
- The provided ONNX model (`skyseg_fp16.onnx`) generates binary masks to distinguish **sky vs. non-sky regions** in videos or camera streams.
- More information about the model can be found at:
https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/tree/1f7811b32b64ddc957269defff84bc87a3f0b74f

---

## Features
- Real-time sky segmentation with **ONNX Runtime**
- GPU acceleration using **CUDAExecutionProvider**
- Optional integration with **TensorRT** (for Jetson devices)
- GPU-accelerated post-processing (thresholding + morphological filters)
- Output video generation showing **original input vs. segmented view**
- Works with both **video files** and **camera input**

---

## Python Implementation - Sky Segmentation

### Dependencies
The Python implementation requires the following packages listed in `requirements.txt`:
- `opencv-python`
- `numpy`
- `onnxruntime-gpu`

### Installation
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Execution
To run the Python sky segmentation implementation:

```bash
python sky_seg.py [video_path]
```

## Command Line Arguments
The Python implementation accepts the following command line arguments:

- `[video_path]` (optional): Path to the input video file.  
  If not provided, the webcam (device `0`) will be used.

---

## Example Usage
Run with webcam:
```bash
python sky_seg.py
```

## Output
- The processed video will be saved in the same directory as the input file, with the suffix `_F16_segmentado`.
- Each frame of the output shows side-by-side:
  - **Original frame**
  - **Segmented view** with colors:
    - Sky → Pink
    - Non-sky → Green
  - Labels showing regions as `"Sky Area: Navigable"` or `"Danger: No-Navigation"`

---

## Notes
- If you do not want to recompile OpenCV to use the CUDA flag, you should use the code available in the old_codes folder. The latest version is: skyseg_teste.py.
- If you are going to use the recompiled CUDA version, remember to uninstall the OpenCV installed via pip, as there may be conflicts.
- The project is configured to use **CUDAExecutionProvider** by default.
- To run on Jetson devices with **TensorRT**, uncomment the block of code related to `TensorrtExecutionProvider`.
- Ensure your **OpenCV CUDA DLLs** are available either in the system `PATH` or by adjusting the line in the script:
  ```python
  os.add_dll_directory(r"C:\LSIIM\sky_seg\opencv_dlls")
 ```
