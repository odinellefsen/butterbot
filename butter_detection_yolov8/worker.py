"""
Vision worker — IPC subprocess for the Rust orchestrator.

On the Raspberry Pi: loads the YOLOv8 ONNX model, opens the Pi camera, and
streams per-frame detection results to the orchestrator.

On Mac (no camera): runs as a stub — signals ready immediately and sits idle.
The orchestrator handles the absence of ButterDetected events gracefully.

Protocol (stdout, newline-delimited JSON):
  {"type": "ready"}
  {"type": "butter_detected", "bbox": {"x1": 120, "y1": 80, "x2": 400, "y2": 310, "confidence": 0.87}}
  {"type": "butter_lost"}
  {"type": "delivery_complete"}
"""

import json
import os
import platform
import sys
import time

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "runs", "detect", "train", "weights", "best.onnx"
)
CONFIDENCE_THRESHOLD = 0.45
INPUT_SIZE = 640

# Fraction of frame area at which the butter is considered "reached".
DELIVERY_AREA_THRESHOLD = 0.25


def emit(msg: dict) -> None:
    print(json.dumps(msg), flush=True)


def is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            return "Raspberry Pi" in f.read()
    except OSError:
        return False


# ── Raspberry Pi: real ONNX + camera pipeline ────────────────────────────────

def run_pi() -> None:
    import cv2
    import numpy as np
    import onnxruntime as rt

    session = rt.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Vision Worker] Camera not available", file=sys.stderr)
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    emit({"type": "ready"})

    butter_was_visible = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = preprocess(frame)
            outputs = session.run(None, {input_name: img})
            boxes = postprocess(outputs, orig_w, orig_h)

            if boxes:
                # Use the highest-confidence detection.
                x1, y1, x2, y2, confidence = max(boxes, key=lambda b: b[4])
                area_fraction = ((x2 - x1) * (y2 - y1)) / (orig_w * orig_h)

                emit({
                    "type": "butter_detected",
                    "bbox": {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": confidence,
                    },
                })

                if area_fraction >= DELIVERY_AREA_THRESHOLD:
                    emit({"type": "delivery_complete"})

                butter_was_visible = True
            else:
                if butter_was_visible:
                    emit({"type": "butter_lost"})
                butter_was_visible = False

    finally:
        cap.release()


def preprocess(frame):
    import cv2
    import numpy as np
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = img.transpose(2, 0, 1)[None]
    return img


def postprocess(outputs, orig_w, orig_h):
    import numpy as np
    import cv2
    predictions = outputs[0].squeeze().T
    scores = predictions[:, 4]
    predictions = predictions[scores > CONFIDENCE_THRESHOLD]
    if len(predictions) == 0:
        return []

    boxes_xywh, score_list = [], []
    boxes_xyxy = []
    for cx, cy, w, h, score in predictions:
        x1 = int((cx - w / 2) * orig_w / INPUT_SIZE)
        y1 = int((cy - h / 2) * orig_h / INPUT_SIZE)
        x2 = int((cx + w / 2) * orig_w / INPUT_SIZE)
        y2 = int((cy + h / 2) * orig_h / INPUT_SIZE)
        boxes_xyxy.append([x1, y1, x2, y2])
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        score_list.append(float(score))

    indices = cv2.dnn.NMSBoxes(boxes_xywh, score_list, CONFIDENCE_THRESHOLD, 0.4)
    return [(*boxes_xyxy[i], score_list[i]) for i in indices]


# ── Mac stub: signal ready, do nothing ───────────────────────────────────────

def run_stub() -> None:
    print("[Vision Worker] Running in stub mode (no camera on this platform)", file=sys.stderr)
    emit({"type": "ready"})
    while True:
        time.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if is_raspberry_pi():
        run_pi()
    else:
        run_stub()
