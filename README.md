# YOLO Hand Tracking Project

## Description

This project implements a YOLO model to detect playing cards in order to help the user play blackjack.

### Logistical Specs

The model was initially trained on 2757 images, with each instance of a card having 50 images on average. The training model uses a YOLOv26s model for 150 epochs with 8 workers, batch size 16, and a patience of 30.

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

- Trains the YOLOv26s model for **150 epochs**

**Note:**
Ensure GPU is enabled (CUDA recommended) & modify settings for user device specs
