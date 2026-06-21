# YOLO Hand Tracking Project

## Description

This project implements a YOLO model to detect playing cards in order to help the user play blackjack.

### Logistical Specs

The model was initially trained on 520 images, with each instance of a card having 10 images. The model was then used to automatically annotate 5200 images, with each instance of a card having 100 images. The model training off of the initial data uses a YOLOv26s model for 100 epochs with 8 workers, batch size 16, and a patience of 20.

The specs of the PC that ran the initial model are as follows:

- CPU: Intel Core i5 13600kf
- GPU: Nvidia RTX 3080 Ti 15G VRAM
- RAM: 48G 3200MHz

## Dependencies

Install required libraries:

```
pip install ultralytics
pip install opencv-python
```

## File Execution Order (If Training New Model)

### 1. `train_initial_model.py`

- Trains the YOLOv26s model for **100 epochs**

**Note:**
Ensure GPU is enabled (CUDA recommended) & modify settings for user device specs
