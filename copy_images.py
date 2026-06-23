import os
import shutil
from collections import defaultdict

source_folder = "images"
dest_folder = "new-data/code/data/images/train"
number_of_images = 50

os.makedirs(dest_folder, exist_ok=True)

cards = defaultdict(list)

# Group files by name
for filename in os.listdir(source_folder):
    name = os.path.splitext(filename)[0]

    if name.upper().startswith("JOKER"):
        continue

    card_label = name.rstrip("0123456789")
    cards[card_label].append(filename)

total_copied = 0

# Copy images into same folder
for card_label, files in cards.items():
    files = sorted(files)[:number_of_images]

    for f in files:
        src = os.path.join(source_folder, f)
        dst = os.path.join(dest_folder, f)

        shutil.copy2(src, dst)

        # Counter
        if total_copied % number_of_images == 0:
            print(f"{total_copied} files copied so far...")

print(f"Done: copied up to {number_of_images} images per card into one folder.")