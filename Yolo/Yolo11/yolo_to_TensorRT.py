from ultralytics import YOLO

model = YOLO(r'Weights\best_fev_2025.pt')

model.export(
    format='engine',
    device=0,
    half=True,
    imgsz=640,
    workspace=4
)

print("Exportação para TensorRT (.engine) concluída!")
print("Verifique o diretório do seu modelo .pt para encontrar o arquivo .engine")