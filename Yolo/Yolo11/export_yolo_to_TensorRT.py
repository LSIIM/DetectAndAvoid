from ultralytics import YOLO

model = YOLO(r'Weights/best_yolo26_maio2026_drone_bird_aircraft2.pt')

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