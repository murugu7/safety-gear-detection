from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # pretrained yolov8 nano model

results = model.train(
    data='datayaml.yaml',   # your dataset config
    epochs=50,
    imgsz=640,
    batch=16,
    project='runs1/train',
    name='exp_person',
    exist_ok=True
)

