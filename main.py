import cv2
from ultralytics import YOLO
import logging

# Turning off logging (floods output)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Loading best performing model
model = YOLO("runs/detect/train/weights/last.pt")


cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        # Run model on current frame then output
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Webcam', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped manually")

finally:
    cap.release()
    cv2.destroyAllWindows()