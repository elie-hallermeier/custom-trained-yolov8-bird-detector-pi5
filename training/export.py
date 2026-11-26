"""Exports the trained weights to the int8 TFLite model that the Raspberry Pi runs (models/best_int8.tflite)."""
from ultralytics import YOLO

TRAINED_WEIGHTS = "runs/detect/train/weights/best.pt"
DATASET = "Bird-Feeder-1/data.yaml"  # images from it calibrate the int8 quantization
IMAGE_SIZE = 320

if __name__ == "__main__":
    model = YOLO(TRAINED_WEIGHTS)
    model.export(format="tflite", int8=True, imgsz=IMAGE_SIZE, data=DATASET)
