from ultralytics import YOLO

model = YOLO(r'Yolo/Yolo11/Weights/best_yolo26_drone_bird_aircraft_junho_2026.pt')

model.export(
    format='engine',
    device=0,
    half=True,
    batch=4,
    imgsz=640,
    dynamic=True
)

print("Exportação para TensorRT (.engine) concluída!")
print("Verifique o diretório do seu modelo .pt para encontrar o arquivo .engine")
