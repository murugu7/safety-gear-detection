from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('D:/AAA/runs/train/exp_finetune/weights/best.pt')

# Open webcam (try 0 or 1 if you have multiple webcams)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Starting webcam detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference on the current frame
    results = model(frame)

    # Annotate frame with detection boxes and labels
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("Cognispectra", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
