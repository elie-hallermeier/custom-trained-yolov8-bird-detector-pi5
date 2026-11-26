"""Trains the bird detector: YOLOv8n fine-tuned on the Bird-Feeder-1 dataset.

These are the settings recorded inside models/best.pt. Everything not set here is the Ultralytics default.
"""
from ultralytics import YOLO

DATASET = "Bird-Feeder-1/data.yaml"
EPOCHS = 50
IMAGE_SIZE = 320  # small on purpose: the model has to run on a Raspberry Pi 5 CPU

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data=DATASET, epochs=EPOCHS, imgsz=IMAGE_SIZE)
