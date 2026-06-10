import cv2
from ultralytics import YOLO
import logging
from collections import defaultdict, deque


# Turning off logging (floods output)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Loading best performing model
model = YOLO("runs/detect/train/weights/best.pt")

suit_names = {
    13: "Diamond",
    14: "Heart",
    15: "Club",
    16: "Spade",
}

ranks = {
    0: "A",
    1: "2",
    2: "3",
    3: "4",
    4: "5",
    5: "6",
    6: "7",
    7: "8",
    8: "9",
    9: "10",
    10: "J",
    11: "Q",
    12: "K"
}

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
        annotated_frame = frame.copy()

        cards = []
        suits = []

        # -----------------------------
        # DETECTION
        # -----------------------------
        for box in results.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < 0.4:
                continue

            x1,y1,x2,y2 = box.xyxy[0].tolist()

            if cls == 17:
                cards.append((x1,y1,x2,y2))

            elif cls in {13,14,15,16}:
                suits.append((cls, x1, y1, x2, y2))

        # -----------------------------
        # CARD PROCESSING
        # -----------------------------
        running_sum = 0
        card_count = len(cards)
        aces = 0
        for card in cards:

            card_x1,card_y1,card_x2,card_y2 = card

            suit_count = 0
            suit_counts = {}

            for suit in suits:

                cls, sx1, sy1, sx2, sy2 = suit

                # Filtering out small suits under number
                box_area = (sx2 - sx1) * (sy2 - sy1)
                card_area = (card_x2 - card_x1) * (card_y2 - card_y1)
                if box_area < 0.01 * card_area:
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
                    suit_counts[cls] = suit_counts.get(cls, 0) + 1
        
            
            # Most occuring suit in card
            card_suit = "Unknown"
            if suit_counts:
                best_cls = max(suit_counts, key=suit_counts.get)
                card_suit = suit_names.get(best_cls, "Unknown")


            # -----------------------------
            # FINAL LABEL
            # -----------------------------
            label = f"{suit_count} of {card_suit}"

            # Soft hand logic (INCOMPLETE)
            if suit_count == 1:
                running_sum += 11
            else:
                running_sum += suit_count
        
            # Only drawing the card
            cv2.rectangle(
                annotated_frame,
                (int(card_x1), int(card_y1)),
                (int(card_x2), int(card_y2)),
                (0, 255, 0),
                2
            )
            
            # Only annotating with suit and value
            cv2.putText(
                annotated_frame,
                label,
                (int(card_x1), int(card_y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        # -----------------------------
        # CARD COUNTING
        # -----------------------------
        if card_count != previous_card_count:
            print(f"Card count changed: {previous_card_count} → {card_count}")
            print("running sum:", running_sum)

        previous_card_count = card_count
        

        # Final frame & waitkey
        cv2.imshow('YOLO Webcam', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()