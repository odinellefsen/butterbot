import cv2
import numpy as np
import onnxruntime as rt
import os
import time

MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs", "detect", "train", "weights", "best.onnx")
CONFIDENCE_THRESHOLD = 0.45
INPUT_SIZE = 640
CLASS_NAMES = ["butter"]


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(outputs, orig_w, orig_h):
    predictions = np.squeeze(outputs[0]).T  # (8400, 5) for 1 class
    scores = predictions[:, 4]
    mask = scores > CONFIDENCE_THRESHOLD
    predictions = predictions[mask]

    if len(predictions) == 0:
        return []

    boxes_xyxy = []
    scores_list = []
    for pred in predictions:
        cx, cy, w, h, score = pred
        x1 = (cx - w / 2) * orig_w / INPUT_SIZE
        y1 = (cy - h / 2) * orig_h / INPUT_SIZE
        x2 = (cx + w / 2) * orig_w / INPUT_SIZE
        y2 = (cy + h / 2) * orig_h / INPUT_SIZE
        boxes_xyxy.append([x1, y1, x2, y2])
        scores_list.append(float(score))

    # Apply NMS to eliminate duplicate boxes around the same object
    indices = cv2.dnn.NMSBoxes(
        [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes_xyxy],
        scores_list,
        CONFIDENCE_THRESHOLD,
        nms_threshold=0.4,
    )

    boxes = []
    for i in indices:
        x1, y1, x2, y2 = boxes_xyxy[i]
        boxes.append((int(x1), int(y1), int(x2), int(y2), scores_list[i]))

    return boxes


def main():
    session = rt.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {orig_w}x{orig_h} — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        img = preprocess(frame)
        outputs = session.run(None, {input_name: img})
        boxes = postprocess(outputs, orig_w, orig_h)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"{elapsed_ms:.1f}ms  {len(boxes)} butter", end="\r")

        for x1, y1, x2, y2, score in boxes:
            label = f"butter {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Butter Detection (ONNX)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
