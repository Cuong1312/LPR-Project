import shutil
from ultralytics import YOLO

# Load YOLOv8n pretrained
model = YOLO('yolov8n.pt')

# Train
model.train(
    data='/content/lpr_project/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov8n_lpr',
    exist_ok=True
)

print("Train xong!")

# Luu ngay ve Drive
shutil.copy(
    '/content/runs/detect/yolov8n_lpr/weights/best.pt',
    '/content/drive/MyDrive/lpr_project/yolov8n_best.pt'
)
shutil.copy(
    '/content/runs/detect/yolov8n_lpr/results.png',
    '/content/drive/MyDrive/lpr_project/yolov8n_results.png'
)

print("Da luu ve Drive:")
print("  - yolov8n_best.pt")
print("  - yolov8n_results.png")