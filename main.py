import cv2
from ultralytics import YOLO
import logging

# Turning off logging (floods output)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Loading best performing model
model = YOLO("runs/detect/train/weights/last.pt")


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

        cards = []
        suits = []

        # Collecting bounding boxes of cards and suits
        for box in results.boxes:
            cls = int(box.cls[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()

            if cls == 17:
                cards.append((x1,y1,x2,y2))
            elif cls in {13,14,15,16}:
                suits.append((cls, x1, y1, x2, y2))

        # Counting the number of suits in the card (determines card value)
        for card in cards:
            card_x1,card_y1,card_x2,card_y2 = card

            suit_count = 0

            for suit in suits:
                _, sx1, sy1, sx2, sy2 = suit

                # Filtering out small suits under number
                box_area = (sx2 - sx1) * (sy2 - sy1)
                if box_area < 600:
                    continue

                # Center of suit
                center_x = (sx1 + sx2) / 2
                center_y = (sy1 + sy2) / 2

                # Finding if suit is in card bounding box
                if (
                    card_x1 <= center_x <= card_x2 and
                    card_y1 <= center_y <= card_y2
                ):
                    suit_count += 1
        
            # Annotating frame
            cv2.putText(
                annotated_frame,
                str(suit_count),
                (int(card_x1), int(card_y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow('YOLO Webcam', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped manually")

finally:
    cap.release()
    cv2.destroyAllWindows()