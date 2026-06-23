import os
import shutil
from collections import defaultdict

source_folder = "images"
dest_folder = "new-data/code/data/images/train"
number_of_images = 20

os.makedirs(dest_folder, exist_ok=True)

cards = defaultdict(list)

# Group by card label (e.g., 2c, 10d)
for filename in os.listdir(source_folder):
    name = os.path.splitext(filename)[0]

    if name.upper().startswith("JOKER"):
        continue

    # remove trailing digits (index like 0,1,2,...)
    card_label = name.rstrip("0123456789")

    cards[card_label].append(filename)

# Copy up to 10 of each card into ONE folder
for card_label, files in cards.items():
    files = sorted(files)[:number_of_images]

    for f in files:
        src = os.path.join(source_folder, f)
        dst = os.path.join(dest_folder, f)

        shutil.copy2(src, dst)

print(f"Done: copied up to {number_of_images} images per card into one folder.")