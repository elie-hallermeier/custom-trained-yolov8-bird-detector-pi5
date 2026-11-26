#!/usr/bin/env python3
"""
Raspberry Pi 5: Bird detector (TFLite YOLOv8) with live Flask MJPEG preview

- Captures frames from Pi Camera (Picamera2)
- Runs TFLite YOLOv8 (int8/uint8/float32 supported)
- Draws rectangles + confidence around detected birds
- Streams live preview at:  http://<pi-ip>:5000/

Requirements (typical):
  sudo apt install -y python3-opencv
  pip install flask numpy picamera2
  pip install tflite-runtime   # recommended on Raspberry Pi
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

try:
    from tflite_runtime.interpreter import Interpreter  # best for Pi
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------- CONFIG ----------------
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "best_int8.tflite")

# Camera resolution (lower = faster)
CAM_W = 640
CAM_H = 480

# Detection thresholds
CONF_THRESH = 0.50
IOU_THRESH = 0.10

# Run inference every N frames (increase to boost FPS)
DETECT_EVERY_N_FRAMES = 1

# TFLite threads (Pi 5 has 4 cores; try 4)
TFLITE_THREADS = 4
# --------------------------------------


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] -> [x1, y1, x2, y2]."""
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1_i = int(max(0, min(w - 1, x1)))
    y1_i = int(max(0, min(h - 1, y1)))
    x2_i = int(max(0, min(w - 1, x2)))
    y2_i = int(max(0, min(h - 1, y2)))
    if x2_i < x1_i:
        x1_i, x2_i = x2_i, x1_i
    if y2_i < y1_i:
        y1_i, y2_i = y2_i, y1_i
    return x1_i, y1_i, x2_i, y2_i


def make_interpreter(model_path: str, threads: int) -> Interpreter:
    """
    Create TFLite interpreter. tflite-runtime supports num_threads directly.
    tensorflow.lite might not, so we fall back safely.
    """
    try:
        return Interpreter(model_path=model_path, num_threads=threads)
    except TypeError:
        return Interpreter(model_path=model_path)


class BirdDetector:
    def __init__(self, model_path: str, conf_thresh: float, iou_thresh: float) -> None:
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.interpreter = make_interpreter(model_path, TFLITE_THREADS)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.in_h = int(self.input_details[0]["shape"][1])
        self.in_w = int(self.input_details[0]["shape"][2])

        self.in_dtype = self.input_details[0]["dtype"]
        self.in_scale, self.in_zero = self._get_quant_params(self.input_details[0])

        self.out_dtype = self.output_details[0]["dtype"]
        self.out_scale, self.out_zero = self._get_quant_params(self.output_details[0])

    @staticmethod
    def _get_quant_params(detail: dict) -> Tuple[float, int]:
        q = detail.get("quantization", (0.0, 0))
        if q is None:
            return 0.0, 0
        scale = float(q[0]) if len(q) > 0 else 0.0
        zero = int(q[1]) if len(q) > 1 else 0
        return scale, zero

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns input tensor shaped (1, in_h, in_w, 3) with proper dtype/quantization.
        """
        img = cv2.resize(frame_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.in_dtype == np.float32:
            img_f = img.astype(np.float32) / 255.0
            return np.expand_dims(img_f, axis=0)

        img_f = img.astype(np.float32)

        if self.in_scale and self.in_scale > 0:
            img_q = img_f / self.in_scale + self.in_zero
        else:
            img_q = img_f

        if self.in_dtype == np.uint8:
            img_q = np.clip(img_q, 0, 255).astype(np.uint8)
        elif self.in_dtype == np.int8:
            img_q = np.clip(img_q, -128, 127).astype(np.int8)
        else:
            img_q = img_q.astype(self.in_dtype)

        return np.expand_dims(img_q, axis=0)

    def _postprocess_output(self, out: np.ndarray) -> np.ndarray:
        """
        Ensures float32 output for decoding.
        Also squeezes and transposes to shape (num_anchors, dims) if needed.
        """
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]

        if out.dtype != np.float32:
            if self.out_scale and self.out_scale > 0:
                out = (out.astype(np.float32) - float(self.out_zero)) * float(self.out_scale)
            else:
                out = out.astype(np.float32)

        if out.ndim != 2:
            out = np.reshape(out, (out.shape[0], -1))

        if out.shape[0] < out.shape[1] and out.shape[0] <= 128:
            out = out.T

        return out.astype(np.float32)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Returns list of detections: (x1, y1, x2, y2, score) in frame coordinates.
        """
        h, w = frame_bgr.shape[:2]

        inp = self._preprocess(frame_bgr)
        self.interpreter.set_tensor(self.input_details[0]["index"], inp)
        self.interpreter.invoke()

        out_raw = self.interpreter.get_tensor(self.output_details[0]["index"])
        out = self._postprocess_output(out_raw)

        if out.shape[1] < 5:
            return []

        boxes = out[:, :4]  # cx, cy, bw, bh
        dims = out.shape[1]

        if dims == 5:
            scores = out[:, 4]
        else:
            scores_all = out[:, 4:]
            scores = scores_all.max(axis=1)

        mask = scores >= self.conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]

        if boxes.size == 0:
            return []

        max_val = float(np.max(boxes))
        if max_val <= 1.5:
            boxes[:, 0] *= w
            boxes[:, 1] *= h
            boxes[:, 2] *= w
            boxes[:, 3] *= h
        else:
            boxes[:, 0] *= w / float(self.in_w)
            boxes[:, 1] *= h / float(self.in_h)
            boxes[:, 2] *= w / float(self.in_w)
            boxes[:, 3] *= h / float(self.in_h)

        boxes_xyxy = xywh_to_xyxy(boxes)

        boxes_xywh = np.zeros_like(boxes_xyxy, dtype=np.float32)
        boxes_xywh[:, 0] = boxes_xyxy[:, 0]
        boxes_xywh[:, 1] = boxes_xyxy[:, 1]
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        b_list = boxes_xywh.tolist()
        s_list = scores.tolist()

        indices = cv2.dnn.NMSBoxes(
            b_list,
            s_list,
            score_threshold=self.conf_thresh,
            nms_threshold=self.iou_thresh,
        )

        detections: List[Tuple[int, int, int, int, float]] = []
        if len(indices) == 0:
            return detections

        for i in np.array(indices).flatten():
            x, y, bw, bh = b_list[i]
            score = float(s_list[i])

            x1, y1, x2, y2 = clamp_box(x, y, x + bw, y + bh, w, h)
            detections.append((x1, y1, x2, y2, score))

        return detections


# ---------------- Pi Camera ----------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size": (CAM_W, CAM_H), "format": "BGR888"},
        buffer_count=1,
    )
)
picam2.start()
time.sleep(1.0)

# ---------------- Detector ----------------
detector = BirdDetector(MODEL_PATH, CONF_THRESH, IOU_THRESH)

# ---------------- Flask UI ----------------
app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <title>Pi Bird Detector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    html, body { margin:0; padding:0; height:100%; width:100%; background:#000; overflow:hidden; }
    img { display:block; width:100%; height:100%; object-fit:contain; }
  </style>
</head>
<body>
  <img src="{{ url_for('video_feed') }}" alt="Video stream">
</body>
</html>
"""


@app.route("/")
def index() -> str:
    return render_template_string(HTML)


def gen_frames() -> bytes:
    frame_idx = 0
    last_detections: List[Tuple[int, int, int, int, float]] = []
    last_conf: float = 0.0

    while True:
        t0 = time.perf_counter()
        try:
            frame = picam2.capture_array()
        except Exception as exc:
            print(f"[WARN] capture_array failed: {exc}")
            time.sleep(0.2)
            continue

        frame_idx += 1
        do_detect = (frame_idx % max(1, DETECT_EVERY_N_FRAMES)) == 0

        if do_detect:
            dets = detector.detect(frame)
            last_detections = dets
            last_conf = max((d[4] for d in dets), default=0.0)

        for (x1, y1, x2, y2, score) in last_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"bird {score:.2f}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        dt = time.perf_counter() - t0
        fps = (1.0 / dt) if dt > 0 else 0.0

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Conf: {last_conf:.2f}  Dets: {len(last_detections)}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed() -> Response:
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        print("Starting bird stream on http://0.0.0.0:5000/")
        app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
    finally:
        try:
            picam2.close()
        except Exception:
            pass
        print("Clean shutdown.")
