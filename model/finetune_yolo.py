"""
Fine-tune YOLO11s on top-down surveillance dataset.

Trains the yolo11s.pt model on data/topDown for improved 
top-down person detection in surveillance scenarios.
"""

from pathlib import Path
from ultralytics import YOLO
import torch


# Get absolute paths relative to this script's location
MODEL_DIR = Path(__file__).parent.resolve()
ROOT = MODEL_DIR.parent

# Configuration
BASE_MODEL = MODEL_DIR / "yolo11s.pt"
DATA_YAML = ROOT / "data" / "topDown" / "data.yaml"

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 10

# Load model
print(f"\nFine-tuning YOLO Training Script")
print(f"\nLoading model: {BASE_MODEL}")
model = YOLO(str(BASE_MODEL))

# Auto-detect device
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Image size: {IMG_SIZE}")

# Train
print("\nTraining:\n")
results = model.tune(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    patience=PATIENCE,
    device=device,
    project="runs/finetune",
    name="yolo11s_topdown",
    verbose=False,    # Reduce output
    single_cls=True,  # Single class detection

    
    
    # Random Data Augmentation
    hsv_h=0.015,      # Hue
    hsv_s=0.4,        # Saturation
    hsv_v=0.4,        # Brightness
    degrees=5.0,      # Rotation
    fliplr=0.5,       # Horizontal Flip

    iterations=50
)

# Validate on test set
print("\nValidation:")
metrics = model.val(split='test')

print("\nTraining Completed:")

print(f"\nmAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"\nBest model: runs/finetune/yolo11s_topdown/weights/best.pt")