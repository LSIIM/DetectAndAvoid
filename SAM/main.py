import os
import time
import urllib

import cv2
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image
import math
import argparse

from sam2.build_sam import build_sam2_object_tracker

import supervision as sv

def resize_mask(mask, video_height, video_width):
    mask = torch.tensor(mask, device='cpu')
    mask = torch.nn.functional.interpolate(mask,
                                            size=(video_height, video_width),
                                            mode="bilinear",
                                            align_corners=False,
                                            )
    return mask

def tratar_first_frame(frame):

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    #edges = cv2.Canny(blurred, 0, 100)
    edges = cv2.Canny(blurred, 100, 135)

    # detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

    best_line = 0
    max_length = 0
    
    if lines.size != 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # calc comp
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  

            #calc ang
            angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * (180 / np.pi)  # 

            #line w low ang and max comp
            if angle < 5 and length > max_length:
                best_line = (x1, y1, x2, y2)
                max_length = length
    
    mask = np.zeros_like(frame, dtype=np.uint8)

    if best_line:
        x1, y1, x2, y2 = best_line

        # Criar um polígono abaixo da linha do horizonte (área a ser preenchida com preto)
        height, width = frame.shape[:2]
        poly_points = np.array([[(0, y1), (width, y2), (width, height), (0, height)]], dtype=np.int32)

        # Preencher com preto a região abaixo da linha do horizonte
        cv2.fillPoly(frame, poly_points, (0, 0, 0))
                
    hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    upper_hue = np.array([120,255,255])
    lower_hue = np.array([90,0,150]) 

    mask      = cv2.inRange(hsv, lower_hue, upper_hue)

    blur = cv2.GaussianBlur(mask, (9, 9), 0)

    _, img_thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

    mask = cv2.morphologyEx(img_thresholded, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    selected_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Ajuste este valor para detectar áreas brancas grandes
            selected_contours.append(contour)
            black_image = np.zeros(mask.shape, np.uint8)
            cv2.fillPoly(black_image, pts=selected_contours, color= (255,255,255))
    
    coord_white = np.column_stack(np.where(black_image == 255))

    if(len(coord_white) > 0):

        y_min, x_min = coord_white.min(axis=0)
        y_max, x_max = coord_white.max(axis=0)
        return np.array([[x_min, y_min], [x_max, y_max]])

    return None

parser = argparse.ArgumentParser(description="Processador de vídeo ou câmera")
parser.add_argument("video", nargs='?', default=None, help="Caminho para o vídeo ou use a câmera se não especificado")
args = parser.parse_args()

VIDEO_STREAM = f"my_files/{args.video}"
NUM_OBJECTS = 1
SAM_CHECKPOINT_FILEPATH = "checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
OUTPUT_PATH = VIDEO_STREAM + "_segmented45.mp4"
DEVICE = 'cuda'

if args.video is None:
    print("Usando a câmera (cv2.VideoCapture(0))")
    video_stream = cv2.VideoCapture(0)
else:
    print(f"Usando o vídeo: {args.video}")
    video_stream = cv2.VideoCapture(VIDEO_STREAM)

if not video_stream.isOpened():
    print("Erro ao abrir vídeo ou câmera.")
    exit()

video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=False
                                )
available_slots = np.inf

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (video_width, video_height))

frame_to_process = 0

first_frame = True

frame_count = 0
start_time = time.time()


with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    while video_stream.isOpened():
        # Get next frame
        ret, frame = video_stream.read()

        # Exit if no frames remaining
        if not ret:
            break

        # Convert frame from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Simulate detection on first frame
        if first_frame:
            
            bbox = tratar_first_frame(frame.copy())

            sam_out = sam.track_new_object(img=img,
                                          box=bbox
                                          )

            first_frame = False

        else:
        
            if(frame_to_process == 45):
              sam_out = sam.track_all_objects(img=img)
              frame_to_process = 0
            else:
              frame_to_process += 1
        

        frame_copy = frame.copy()
        frame_copy = cv2.resize(frame, (video_width, video_height))

        mask =  resize_mask(sam_out['pred_masks'], video_height, video_width)
        mask = (mask > 0.0).numpy()

        mask_center = mask[0, 0]  # Pega a primeira máscara do batch e primeiro canal
        bg_mask = ~mask_center
            
        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]

            frame_copy[:, :] = [144, 238, 144]

            frame_copy[obj_mask] = [255, 105, 180]
        
        y_indices, x_indices = np.where(mask_center)
        
        if len(y_indices) > 0:
            center_y = int(round(np.mean(y_indices)))
            center_x = int(round(np.mean(x_indices)))
            cv2.putText(frame_copy, "Sky Area: Navigable Zone", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        y_bg, x_bg = np.where(bg_mask)

        if len(y_bg):
            center_y_bg = int(round(np.mean(y_bg)))
            center_x_bg = int(round(np.mean(x_bg)))
            cv2.putText(frame_copy, "Danger Area: No-Navigation Zone", (center_x_bg, center_y_bg), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        rgb_frame = Image.fromarray(frame_copy)
        
        clear_output(wait=True)
        display(rgb_frame)
        
        img = np.array(rgb_frame)  
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

        out.write(img)
        frame_count += 1

elapsed_time = time.time() - start_time
avg_fps = frame_count / elapsed_time
print(f"Avg FPS: {int(avg_fps)}")



video_stream.release()


