from ultralytics import YOLO
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

# Paths
model_path = 'D:/AAA/runs/train/exp_finetune/weights/best.pt'
images_dir = 'D:/AAA/Dataset/images/test'
labels_dir = 'D:/AAA/Dataset/labels/test'  # Ground truth YOLO label files
results_dir = 'D:/AAA/results'
debug_dir = 'D:/AAA/debug_vis'

# Clean/Create directories
for dir_path in [results_dir, debug_dir]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

# Classes
class_names = ['helmet', 'apron', 'gloves', 'no helmet', 'no apron', 'no gloves']
num_classes = len(class_names)

# Load YOLO model
model = YOLO(model_path)

# IoU Threshold
IOU_THRESHOLD = 0.5

# Helper: Compute IoU
def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# Metrics storage
y_true = []
y_pred = []

# Supported image formats
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']

# Process each test image
image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in supported_formats]

for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)
    label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h

                gt_boxes.append((class_id, (x1, y1, x2, y2)))

    # Run YOLO Prediction
    results = model(image_path)
    predictions = results[0].boxes.xyxy.cpu().numpy()
    pred_classes = results[0].boxes.cls.cpu().numpy()

    matched_gt = set()
    matched_preds = set()

    # Matching predictions to ground truths
    for pred_idx, (pred_box, pred_class) in enumerate(zip(predictions, pred_classes)):
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, (gt_class, gt_box) in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou >= IOU_THRESHOLD and iou > best_iou and gt_idx not in matched_gt:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            y_true.append(gt_boxes[best_gt_idx][0])
            y_pred.append(int(pred_class))
            matched_gt.add(best_gt_idx)
            matched_preds.add(pred_idx)
        else:
            # Predicted something that does not match GT -> false positive
            y_true.append(num_classes)  # background class index (6)
            y_pred.append(int(pred_class))

    # False Negatives: GT boxes with no matching prediction
    for gt_idx, (gt_class, _) in enumerate(gt_boxes):
        if gt_idx not in matched_gt:
            y_true.append(gt_class)
            y_pred.append(num_classes)  # background class index (6)

    # Visualization of GT and predictions
    for idx, (gt_class, gt_box) in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, gt_box)
        color = (0, 255, 0) if idx in matched_gt else (0, 0, 255)  # Green if matched else Red
        label = f"GT: {class_names[gt_class]}" if idx in matched_gt else f"Missed: {class_names[gt_class]}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for idx, (pred_box, pred_class) in enumerate(zip(predictions, pred_classes)):
        x1, y1, x2, y2 = map(int, pred_box)
        if idx in matched_preds:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan for matched preds
            cv2.putText(img, f"Pred: {class_names[int(pred_class)]}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for false positive preds
            cv2.putText(img, f"FP: {class_names[int(pred_class)]}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    debug_image_path = os.path.join(debug_dir, image_file)
    cv2.imwrite(debug_image_path, img)

    # Save prediction image (annotated by ultralytics)
    result_image_path = os.path.join(results_dir, image_file)
    annotated_frame = results[0].plot()
    cv2.imwrite(result_image_path, annotated_frame)

    print(f"Saved result: {result_image_path}")
    print(f"Debug image saved: {debug_image_path}")

# Add background class for metrics
class_names_with_bg = class_names + ['background']

# Classification report
report_text = classification_report(y_true, y_pred, target_names=class_names_with_bg, zero_division=0)
print("\nClassification Report:")
print(report_text)

# Save Classification Report as Image
plt.figure(figsize=(8, 4))  
plt.axis('off')
plt.text(0.02, 0.98, report_text, fontsize=14, family='monospace', verticalalignment='top')
plt.title('Classification Report', pad=20, fontsize=16)
plt.savefig('classification_report.png', dpi=300, bbox_inches='tight')
plt.show()
print("Classification Report saved as 'classification_report.png'")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names_with_bg, yticklabels=class_names_with_bg)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# Precision, Recall, F1-Score Bar Chart
report = classification_report(y_true, y_pred, target_names=class_names_with_bg, output_dict=True, zero_division=0)
metrics_df = pd.DataFrame(report).T[:-1]  # Exclude 'accuracy' row

plt.figure(figsize=(10, 6))
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title('Evaluation Metrics per Class')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('metrics_bar_chart.png', dpi=300)
plt.show()

print("Confusion Matrix, Classification Report, and Metrics Bar Chart saved.")
