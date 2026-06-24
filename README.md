# YOLO Blackjack Helper

## Description

This project implements a YOLO model to detect playing cards in order to help the user play Blackjack.

## Technical Outline

### Initial Model

The initial YOLO model is a YOLOv26m trained on 1040 images, with each card having 20 instances. The model was trained for 150 epochs, with a patience of 30, 8 workers, and a batch size of 16.

### Improved Model

The initial YOLO model was used to annotate a larger dataset of 2600 images, with each card having 50 instances. The improved model was trained for 200 epochs, with a patience of 40, 8 workers, and a batch size of 16.

### Device Specs

The specs of the PC that ran the initial model are as follows:

- CPU: Intel Core i5 13600kf
- GPU: Nvidia RTX 3080 Ti 12G VRAM
- RAM: 48G 3200MHz

## CAUTION

- The `images` folder contains 2757 images
- The `training-data` folder contains 1040 images and 1040 label files
- The `new-data` folder contains 2600 images, 2600 annotated images, and 2600 label files

User discretion is advised when opening.

## Dependencies

Install required libraries:

```
pip install opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
```

## File Execution Order (If Training New Model)

### 1. `train_initial_model.py`

- Trains the YOLOv26s model for **150 epochs**

### 2. `copy_images.py`

- Copies images from `images` to `new-data/code/data/images/train`

**Note:** amount of images used can be adjusted

### 3. `annotate.py`

- Annotates the images from `new-data/code/data/images/train` and creates annotations in `new-data/code/data/results/train` and labels in `new-data/code/data/labels/train`

### 4. `val.py`

- Creates a validation folder using 20% of the training data

**Note:** validation size can be adjusted

### 5. `train_new_mode.py`

- Trains the new YOLO model for **200 epochs**

**Note:** directories must be reconfigured in `config.yaml`

### 6. `main.py`

- Final Blackjack helped program

**Note:**
Ensure GPU is enabled (CUDA recommended) & modify settings for user device specs
