from ultralytics import YOLO

# Choose the lightest model variant for efficiency
model = YOLO("yolov8n.yaml")  # Switch to YOLOv8 Nano for better performance on Raspberry Pi

# Use the model with adjusted training options
model.train(
    data="config.yaml",
    epochs=50,  # Start with fewer epochs
    batch=32,  # Increase batch size
    auto_augment='randaugment',
    patience=10  # Early stopping with patience of 10 epochs
)