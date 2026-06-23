import cv2
from ultralytics import YOLO

# Loading best performing model
model = YOLO("runs/train/weights/best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
previous_card_count = 0
count = 0

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
        results = model(frame, conf=0.4, verbose=False)[0]
        annotated_frame = results.plot()

        # Reprocess if number of cards changes
        if len(results.boxes) != previous_card_count:

            visible_sum = 0
            num_aces = 0

            # Printing all visible cards if there is a change
            for card in results.boxes:
                class_id = int(card.cls[0])
                class_name = int(model.names[class_id])

                if class_name == 1: 
                    num_aces += 1

                # Sum of cards visible
                visible_sum += class_name

            # Ace logic
            if (visible_sum <= 10):
                visible_sum += (num_aces * 10)

            # Base hit/stand logic
            if (
                (num_aces < 1 and visible_sum == 8) or 
                (visible_sum < 17 and num_aces > 0)
            ):
                print("HIT")

            elif (visible_sum >= 17 and num_aces == 0):
                print("STAND")

                
                
                
            print("Sum: ", visible_sum)

            # Update previous count
            previous_card_count = len(results.boxes)

        # Final frame & waitkey
        cv2.imshow('YOLO Webcam', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()