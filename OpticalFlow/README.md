# Optical Flow Project

## Overview
This project implements Optical Flow in and Python using OpenCV.

## Features
- Computes dense and sparse optical flow
- Supports Lucas-Kanade and Farneback methods
- Python implementation with machine learning clustering

## Python Implementation - Optical Flow

### Dependencies
The Python implementation requires the following packages listed in `requirements.txt`:
- scikit-learn
- opencv-python
- numpy

### Installation
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Execution
To run the Python optical flow implementation:

```bash
python opticalflow.py <clusters> [video_path]
```

### Command Line Arguments
The Python implementation accepts the following command line arguments:

- `<clusters>` (required): Number of clusters for fuzzy c-means clustering
- `[video_path]` (optional): Path to the input video file

Example usage:
```bash
python opticalflow.py 5
python opticalflow.py 3 path/to/video.mp4
```
