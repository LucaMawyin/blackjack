import os
import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/train/weights/best.pt")

root_dir = "new-data/code/data/" 

image_folder = root_dir + "images/train"
label_folder = root_dir + "labels/train"
annotated_folder = root_dir + "results/train"

os.makedirs(label_folder, exist_ok=True)
os.makedirs(annotated_folder, exist_ok=True)

label_count = 0

for img_name in os.listdir(image_folder):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_folder, img_name)

    results = model(img_path, conf=0.25, verbose=False)[0]

    # Images with no detection
    if results.boxes is None or len(results.boxes) == 0:
        print(f"No annotations: {img_name}")

    # Save labels
    label_path = os.path.join(
        label_folder,
        os.path.splitext(img_name)[0] + ".txt"
    )

    with open(label_path, "w") as f:
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls = int(box.cls[0])

                # YOLO gives normalized xywh already here:
                x_center, y_center, width, height = box.xywhn[0]

                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
                
                label_count += 1

                if label_count % 100 == 0:
                    print(f"{label_count} labels written")

    annotated_img = results.plot()

    save_path = os.path.join(annotated_folder, img_name)
    cv2.imwrite(save_path, annotated_img)

print("Done auto-annotating")