import cv2
import supervision as sv
from PIL import Image
import numpy as np
import time
from rfdetr import RFDETRBase

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Usando dispositivo:', device)


model = RFDETRBase(pretrain_weights="Weights\checkpoint.pth")




def process_video(video_path, output_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    box_annotator = sv.BoxAnnotator()
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}", end="\r")
        
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        detections = model.predict(pil_image, threshold=threshold)
        
        annotated_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR).copy()
        
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        
        cv2.imshow("RFDETR Detection", annotated_frame)
        out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time
    
    print(f"\nProcessing completed! Total frames processed: {frame_count}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    input_video = "Raw_Videos/fev_corte_3.mp4"
    output_video = "video_rf_detr.mp4"
    
    process_video(input_video, output_video, threshold=0.5)