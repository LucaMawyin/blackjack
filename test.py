import cv2
from ultralytics import YOLO

# Loading best performing model
model = YOLO("runs/card_train_init-2/weights/best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
previous_card_count = 0

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
        results = model(frame, verbose=False)[0]
        annotated_frame = results.plot()

        # Final frame & waitkey
        cv2.imshow('YOLO Webcam', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()