#!/usr/bin/env python3
import time
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# -------- CONFIG --------
MODEL_PATH = str(Path(__file__).parent.parent / "models" / "best_int8.tflite")
CONF_THRESH = 0.10
IOU_THRESH = 0.1
CAM_W, CAM_H = 640, 480
# ------------------------

# letterbox helper
def letterbox_bgr_to_rgb_float(frame_bgr, new_w, new_h, color=(114, 114, 114)):
    h, w = frame_bgr.shape[:2]
    r = min(new_w / w, new_h / h)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    resized = cv2.resize(frame_bgr, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - new_unpad_w
    pad_h = new_h - new_unpad_h
    left = pad_w // 2
    top = pad_h // 2
    padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                cv2.BORDER_CONSTANT, value=color)

    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, r, left, top

def xywh_to_xyxy(boxes):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)

# --- Load TFLite model ---
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
in_h = input_details[0]["shape"][1]
in_w = input_details[0]["shape"][2]
print("model input:", in_w, in_h)

print("INPUT:", input_details[0]["dtype"], "quant:", input_details[0].get("quantization"))
print("OUTPUT:", output_details[0]["dtype"], "quant:", output_details[0].get("quantization"))


# --- Pi Camera setup ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (CAM_W, CAM_H), "format": "BGR888"})
picam2.configure(config)
picam2.start()
time.sleep(1)  # warmup

# --- Flask setup ---
app = Flask(__name__)
HTML = """
<!doctype html>
<html>
<head><title>Pi Bird Detector</title></head>
<body style="margin:0;overflow:hidden;">
  <img src="{{ url_for('video_feed') }}">
</body></html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

def gen_frames():
    while True:
        start = time.perf_counter()

        frame = picam2.capture_array()
        h, w, _ = frame.shape

        img, r, pad_x, pad_y = letterbox_bgr_to_rgb_float(frame, in_w, in_h)

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        if out.ndim == 3:
            out = out[0]
        out = out.T

        boxes = out[:, :4]
        scores_all = out[:, 4:]
        scores = scores_all.max(axis=1)
        mask = scores >= CONF_THRESH
        boxes = boxes[mask]
        scores = scores[mask]

        if boxes.size > 0:
            boxes[:, 0] *= in_w
            boxes[:, 1] *= in_h
            boxes[:, 2] *= in_w
            boxes[:, 3] *= in_h
            boxes_xyxy = xywh_to_xyxy(boxes)

            # unletterbox transform
            boxes_xyxy[:, 0] -= pad_x
            boxes_xyxy[:, 1] -= pad_y
            boxes_xyxy[:, 2] -= pad_x
            boxes_xyxy[:, 3] -= pad_y
            boxes_xyxy /= r

            boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w - 1)
            boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h - 1)
            boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w - 1)
            boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h - 1)

            # build lists for NMS
            boxes_xywh = np.zeros_like(boxes_xyxy)
            boxes_xywh[:, 0] = boxes_xyxy[:, 0]
            boxes_xywh[:, 1] = boxes_xyxy[:, 1]
            boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
            boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

            b_list = boxes_xywh.tolist()
            s_list = scores.tolist()
            indices = cv2.dnn.NMSBoxes(b_list, s_list, CONF_THRESH, IOU_THRESH)

            for i in np.array(indices).flatten() if len(indices) else []:
                x, y, bw, bh = b_list[i]
                score = s_list[i]
                x1, y1 = int(max(x, 0)), int(max(y, 0))
                x2, y2 = int(min(x + bw, w - 1)), int(min(y + bh, h - 1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"bird {score:.2f}", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = 1.0 / max(1e-6, (time.perf_counter() - start))
        cv2.putText(frame, f"FPS: {fps:.1f}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting Flask server at http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
