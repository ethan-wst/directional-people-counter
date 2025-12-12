# Directional People Counter

This repository contains a real-time directional people counting implementation.

This work was carried out as a final project in TXST CS4337 Computer Vision with the intention of implementing a custom IoU based tracking algorithm and full YOLO finetuning and deployment pipeline.

## Repository Contents

### Core Application
- `src/counter.py` — command-line interface for video processing with configurable parameters
- `src/counter_gui.py` — interactive GUI application for real-time visualization
- `src/utils/person_tracker.py` — YOLO-based detection with custom IoU tracking algorithm
- `src/utils/directional_counter.py` — barrier crossing detection logic with counting support
- `src/utils/helpers.py` — core data structures (Track, TrackedPerson, FrameDetections) and visualization utilities
- `src/utils/xgtf_parser.py` — parser for MIVIA ground truth annotations in XGTF format

### Model Training
- `models/finetune_yolo.py` — training script for fine-tuning YOLOv11s on topDown dataset
- `models/ft_yolo11s.pt` — fine-tuned model optimized for top-down surveillance perspectives

### Datasets
- `data/topDown/` — Roboflow dataset for finetuning
- `data/mivia/` — MIVIA People Counting evaluation dataset
  - `config.json` — dataset metadata and configuration
  - `config.txt` — human readable metadata, explaination of crossing configurations

### Supporting Files
- `requirements.txt` — Python dependencies (ultralytics, opencv-python, numpy, Pillow, torch)
- `README.md` — comprehensive documentation for setup, usage, training, and troubleshooting

## Installation

1. **Create virtual environment**
 
    Use either `venv` or `conda` to manage dependencies and isolate your project.


2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install datasets for finetuning**  

    Install the MIVIA People Counting and topDown datasets from the links below:

    - [MIVIA](https://mivia.unisa.it/) dataset: Download from the official site and extract to `data/mivia/`
    - [Roboflow topDown dataset](https://universe.roboflow.com/topdownsurveillance/topdown_surveillance): Download and extract to `data/topDown/`

    _Note: A single video and corresponding ground truth file can be downloaded for fundamental reproducibility: [link](https://drive.google.com/drive/folders/11jdUQuMsqB34q4jhdLAI1n3mKBq0Sq9K?usp=sharing)_

## Running the Application

### GUI Application

Launch the interactive GUI for visual demonstrations and parameter tuning:

```bash
python src/counter_gui.py
```

**Controls:**
- Select video file
- Adjust line position (slider: 0.1-0.9)
- Choose orientation (horizontal/vertical)
- Set count direction (up/down/left/right)
- Configure minimum track age (1-20 frames)
- View real-time statistics and overlays

### Command-Line Interface

Process videos from the command line for batch processing:

```bash
# Basic usage (uses default video)
python src/counter.py

# Specify video file
python src/counter.py path/to/video.mkv
```
### Fine-Tuning YOLOv11s

Run the training script with optimized hyperparameters:

```bash
python models/finetune_yolo.py
```

**Configuration:**
- **Base Model**: YOLOv11s (pretrained on COCO)
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 16
- **Image Size**: 640x640
- **Device**: Auto-detect GPU/CPU
- **Augmentation**: Mosaic, HSV, flip, scale, translate

### Tracker Configuration

The custom IoU-based tracker can be tuned in `src/utils/person_tracker.py`:

```python
tracker = PersonTracker(
    iou_threshold=0.5,        # Minimum IoU for track matching
    max_misses=30,            # Frames before removing lost track
    min_hits=3,               # Frames to confirm track
    new_track_thresh=0.6,     # Confidence to create new track
    confidence_threshold=0.4  # Minimum detection confidence
)
```

**Key Parameters:**
- **`iou_threshold`**: Lower (0.3) = more lenient matching, higher (0.5) = stricter
- **`max_misses`**: Higher values (60-90) help through occlusions
- **`new_track_thresh`**: Higher (0.7-0.8) prevents track fragmentation
- **`min_hits`**: Higher (5-7) filters spurious detections


## Acknowledgments

- **Ultralytics** - YOLOv11 implementation
- **MIVIA** - Ground truth dataset for evaluation
