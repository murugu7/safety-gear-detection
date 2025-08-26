from ultralytics import YOLO

# Load your previously trained model instead of yolov8n.pt
model = YOLO('runs1/train/exp_person/weights/best.pt')

# Fine-tune with the updated dataset
results = model.train(
    data='datayaml.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='runs/train',
    name='exp_finetune',
    exist_ok=True
)
