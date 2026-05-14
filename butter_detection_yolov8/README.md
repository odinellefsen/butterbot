# butter_detection_yolov8

Python vision worker for the Butterbot orchestrator. Trains and runs a custom YOLOv8
model to detect butter in real time from the Pi camera feed. Emits bounding box
coordinates to the orchestrator over stdout JSON.

## Files

| File | Description |
|---|---|
| `worker.py` | IPC subprocess used by the Rust orchestrator. Runs the real ONNX pipeline on Pi, stubs silently on Mac. |
| `main.py` | Trains a YOLOv8 model on the butter dataset. |
| `run_video_onnx.py` | Standalone script to test ONNX inference from a webcam feed. |
| `run_video.py` | Standalone script to test inference using the `.pt` model directly. |
| `config.yaml` | YOLOv8 dataset config — update the `path` to your local dataset. |
| `runs/detect/train/weights/best.onnx` | Trained ONNX model (gitignored). |
| `runs/detect/train/weights/best.pt` | Trained PyTorch model (gitignored). |

## IPC protocol (`worker.py`)

**Stdout (worker → orchestrator):**
```json
{"type": "ready"}
{"type": "butter_detected", "bbox": {"x1": 120, "y1": 80, "x2": 400, "y2": 310, "confidence": 0.87}}
{"type": "butter_lost"}
{"type": "delivery_complete"}
```

`delivery_complete` is emitted when the butter bounding box fills more than 25% of the
frame — used as a proximity proxy for "robot has reached the butter".

On non-Pi platforms (no camera) the worker emits `ready` and then sits idle indefinitely.
The orchestrator handles the absence of `butter_detected` events correctly.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install onnxruntime opencv-python
```

On Raspberry Pi OS, use the CPU-only ONNX runtime:

```bash
pip install onnxruntime  # CPU build is default on ARM
```

## Dataset

Collect images of butter in your environment and organise them as:

```
butter_photos/
├── images/
│   └── train/      ← JPEG/PNG images
└── labels/
    └── train/      ← YOLO-format .txt annotation files
```

Update `config.yaml` to point `path` at your local `butter_photos/` directory.

Annotation format (one line per object):
```
0 <cx> <cy> <width> <height>   # values normalised 0–1, class 0 = butter
```

Use [Label Studio](https://labelstud.io) or [Roboflow](https://roboflow.com) to annotate.

## Train

```bash
python main.py
```

Trains YOLOv8 Nano for 50 epochs. Saves weights to `runs/detect/train/weights/`.

## Export to ONNX

The `worker.py` uses the ONNX model for inference (faster on CPU, no ultralytics dependency
at runtime). Export after training:

```bash
python -c "from ultralytics import YOLO; YOLO('runs/detect/train/weights/best.pt').export(format='onnx')"
```

## Test standalone

```bash
python run_video_onnx.py   # uses ONNX model + webcam
python run_video.py        # uses .pt model + webcam
```

Press `Q` to quit.

## Gitignored

| Path | Reason |
|---|---|
| `runs/detect/train/weights/best.pt` | PyTorch weights — train locally |
| `runs/detect/train/weights/best.onnx` | ONNX weights — exported from best.pt |

After cloning, train the model and export to ONNX before the vision worker will function
on the Pi.
