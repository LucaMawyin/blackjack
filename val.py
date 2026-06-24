import os
import random
import shutil

# Configuration
split_ratio = 0.2  # 20% validation set

root_dir = "new-data/code/data/"

train_images = root_dir + "images/train"
train_labels = root_dir + "labels/train"

val_images = root_dir + "images/val"
val_labels = root_dir + "labels/val"

# Create validation folders if they don't exist
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Get image files
image_files = [
    f for f in os.listdir(train_images)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Shuffle and select validation images
random.shuffle(image_files)
num_val = int(len(image_files) * split_ratio)
val_files = image_files[:num_val]

print(f"Copying {num_val} images to validation set...")

for img_file in val_files:
    # Copy image
    src_img = os.path.join(train_images, img_file)
    dst_img = os.path.join(val_images, img_file)
    shutil.copy2(src_img, dst_img)

    # Copy matching label
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label = os.path.join(train_labels, label_file)

    if os.path.exists(src_label):
        dst_label = os.path.join(val_labels, label_file)
        shutil.copy2(src_label, dst_label)

print("Validation set created.")