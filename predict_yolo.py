from ultralytics import YOLO
import cv2
import os
import shutil

# Load the trained YOLOv8 model
model = YOLO('D:/AAA/runs/train/exp_person/weights/best.pt')

# Paths
test_dir = 'D:/AAA/Dataset/images/test'
output_dir = 'D:/AAA/op'

# Clean/Create output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Supported image formats
supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']

# Iterate through test images
for img_file in os.listdir(test_dir):
    if os.path.splitext(img_file)[1].lower() not in supported_formats:
        continue  # Skip unsupported files

    img_path = os.path.join(test_dir, img_file)

    # Run inference
    op = model(img_path)

    # Visualize predictions on image
    annotated_frame = results[0].plot()

    # Save the annotated image
    save_path = os.path.join(output_dir, img_file)
    cv2.imwrite(save_path, annotated_frame)

    print(f"Saved result: {save_path}")

print("All detections saved successfully.")
