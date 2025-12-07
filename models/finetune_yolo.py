"""Fine-tune YOLO11s on top-down  dataset."""

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
EPOCHS = 100 
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 10  # Increased patience for longer training

# Optimized hyperparameters from ultralytics tuning run
TUNED_HYPERPARAMS = {
    # Learning rate and optimizer
    'lr0': 0.01096,
    'lrf': 0.01194,
    'momentum': 0.93015,
    'weight_decay': 0.00068,
    'warmup_epochs': 3.75313,
    'warmup_momentum': 0.53378,
    
    # Loss components
    'box': 4.10329,
    'cls': 0.35551,
    'dfl': 1.26803,
    
    # Color augmentation
    'hsv_h': 0.02095,
    'hsv_s': 0.29224,
    'hsv_v': 0.8396,
    
    # Geometric augmentation
    'degrees': 6.78525,
    'translate': 0.14314,
    'scale': 0.42724,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.81091,
    
    # Advanced augmentation
    'bgr': 0.0,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0,
    'close_mosaic': 9,  # Must be int, not float
}

# Load model
print(f"\nFine-tuning YOLO Training Script")
print(f"\nLoading model: {BASE_MODEL}")
model = YOLO(str(BASE_MODEL))

# Auto-detect device
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Image size: {IMG_SIZE}")

# Train with optimized hyperparameters
print("\nStarting training with tuned hyperparameters...\n")
results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    patience=PATIENCE,
    device=device,
    project="runs/train",
    name="yolo11s_topdown_tuned",
    single_cls=True,
    save=True,
    save_period=10,  # Save checkpoint every 10 epochs
    **TUNED_HYPERPARAMS  # Unpack optimized hyperparameters
)

# Validate on test set
print("\nValidation:")
metrics = model.val(split='test')

print("\nTraining Completed:")

print(f"\nmAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"\nBest model: ./ft_yolo11s.pt")