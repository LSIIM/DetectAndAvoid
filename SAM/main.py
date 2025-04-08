import os
import time
import urllib

import cv2
import numpy as np
import torch
from IPython.display import clear_output, display
from PIL import Image

from sam2.build_sam import build_sam2_object_tracker
import supervision as sv

class Visualizer:
    def __init__(self,
                 video_width,
                 video_height,
                 ):

        self.video_width = video_width
        self.video_height = video_height

    def resize_mask(self, mask):
        mask = torch.tensor(mask, device='cpu')
        mask = torch.nn.functional.interpolate(mask,
                                               size=(self.video_height, self.video_width),
                                               mode="bilinear",
                                               align_corners=False,
                                               )

        return mask

    def add_frame(self, frame, mask):
        frame = frame.copy()
        frame = cv2.resize(frame, (self.video_width, self.video_height))

        mask = self.resize_mask(mask=mask)
        mask = (mask > 0.0).numpy()

        colors = [[255, 20, 147], [255, 99, 71], [1, 255, 20], [255, 215, 0], [0, 226, 255], [255, 0, 43], [128, 128, 128], [0, 10, 64], [92, 247, 107], [221, 194, 255]]

        for i in range(mask.shape[0]):
            obj_mask = mask[i, 0, :, :]
            frame[obj_mask] = [255, 105, 180]



        rgb_frame = Image.fromarray(frame)
        clear_output(wait=True)
        display(rgb_frame)
        img = np.array(rgb_frame)  # Converte para NumPy
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Converte de RGB para BGR
        #cv2.imshow("Output", img)
        #print(rgb_frame)
        return img


VIDEO_STREAM = f"my_files/vid_dr03.mp4"
YOLO_CHECKPOINT_FILEPATH = "yolov8x-seg.pt"
NUM_OBJECTS = 1
SAM_CHECKPOINT_FILEPATH = "checkpoints/sam2.1_hiera_base_plus.pt"
SAM_CONFIG_FILEPATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
OUTPUT_PATH = VIDEO_STREAM + "_segmented.mp4"
DEVICE = 'cuda'

# Open Video Stream
video_stream = cv2.VideoCapture(VIDEO_STREAM)

video_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))

# For real-time visualization
visualizer = Visualizer(video_width=video_width,
                        video_height=video_height
                        )

sam = build_sam2_object_tracker(num_objects=NUM_OBJECTS,
                                config_file=SAM_CONFIG_FILEPATH,
                                ckpt_path=SAM_CHECKPOINT_FILEPATH,
                                device=DEVICE,
                                verbose=False
                                )
available_slots = np.inf

video_info = sv.VideoInfo.from_video_path(f"my_files/vid_dr03.mp4")

k = 0

first_frame = True
with sv.VideoSink(OUTPUT_PATH, video_info=video_info) as sink:
  with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
      while video_stream.isOpened():
          start_time = time.time()
          # Get next frame
          ret, frame = video_stream.read()

          # Exit if no frames remaining
          if not ret:
              break

          # Convert frame from BGR to RGB
          img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

          # Simulate detection on first frame
          if first_frame:
              bbox = np.array([[[0, 0], [1300, 350]]
                             ]
                             )
              #x_min, y_min, x_max, y_max = tratar_first_frame(img)

              #bbox = np.array([[[x_min, y_min], [x_max, y_max]]
                              #]
                              #)

              sam_out = sam.track_new_object(img=img,
                                            box=bbox
                                            )

              first_frame = False

          else:
            sam_out = sam.track_all_objects(img=img)
            """
              if(k == 10):
                sam_out = sam.track_all_objects(img=img)
                k = 0
              else:
                k += 1
            """
          final = visualizer.add_frame(frame=frame, mask=sam_out['pred_masks'])
          sink.write_frame(final)

video_stream.release()