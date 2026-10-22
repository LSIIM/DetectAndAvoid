import sys
import time
import cv2
import numpy as np

from modules.yoloModule import YOLODetector
from modules.opticalFlowModuleCPU import OpticalFlowTracker
from modules.avoidDirection import CollisionAvoidanceAnalyzer

YOLO_MODEL_PATH = r"./modules/bestYOLO11n.pt"
TRACKER_CONFIG = "bytetrack.yaml"

def main():

    videoPath = r"./videos/multiplosdronesclean.mp4"
    
    cap = cv2.VideoCapture(videoPath)

    if not cap.isOpened():
        print(f"Não foi possível abrir o vídeo em: {videoPath}")
        return 1
        
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30
    
    resizeHeight = 480
    scale = resizeHeight / frameHeight
    processingWidth = int(frameWidth * scale)
    processingHeight = resizeHeight

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('resultado.mp4', fourcc, fps, (processingWidth, processingHeight))
    
    yoloDetector = YOLODetector(
        modelPath = YOLO_MODEL_PATH,
        trackerConfig = TRACKER_CONFIG,
        confidenceThreshold = 0.25
    )
    
    tracker = OpticalFlowTracker(
        fps=fps,
        processingSize=(processingWidth, processingHeight)
    )

    collision = CollisionAvoidanceAnalyzer(
        clusters=5,
        fps=fps
    )

    frameCount = 0

    while True:

        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Fim do vídeo.")
            break
            
        frameStart = time.time()
        frameCount += 1

        frameResized = cv2.resize(frame, (processingWidth, processingHeight))
        
        boxes = yoloDetector.processFrame(frameResized)
        
        detectionMask = np.zeros((processingHeight, processingWidth), dtype=np.uint8)
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(detectionMask, (x1, y1), (x2, y2), 255, -1)

        resultFrame, trackingData = tracker.processFrame(frameResized, detectionMask)
        
        hudFrame = collision.analyzeAndDraw(resultFrame, trackingData, boxes)

        timeSec = time.time() - frameStart
        currentFps = 1.0 / timeSec if timeSec > 0 else 0.0
        
        infoText = f"Frame: {frameCount} | FPS: {currentFps:.1f}"
        cv2.putText(hudFrame, infoText, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(hudFrame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())