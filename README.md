# Manufacturing Defect Classification

A hierarchical deep learning system for multi-domain manufacturing defect detection using PyTorch.

## Features

- **Domain Classification**: Routes images to correct specialist (30 industrial domains)
- **Specialist Models**: Per-domain defect classification using MSANet architecture
- **Auto-detection**: Handles various folder structures (flat, train/val split)
- **Early Stopping**: Prevents overfitting with configurable patience
- **Resume Training**: Automatically skips already-trained domains

## Architecture

```
Input Image → Domain Classifier → Specialist Model → Defect Prediction
                  (30 classes)      (domain-specific)
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/manufacturing-defect-classification.git
cd manufacturing-defect-classification
pip install -r requirements.txt
```

## Project Structure

```
├── models/              # Model architectures
├── datasets/            # Training data (not in repo)
├── checkpoints/         # Saved models (not in repo)
├── config.py           # Configuration
├── train_*.py          # Training scripts
└── test_pipeline.py    # Inference pipeline
```

## Usage

### Train Domain Classifier
```bash
python train_domain_classifier.py
```

### Train All Specialists
```bash
python train_batch.py
```

### Train Specific Domain
```bash
python train_batch.py concrete
```

### Test Pipeline
```bash
python test_pipeline.py --mode full --image sample.jpg
```

## Training Status

| Component | Status | Accuracy |
|-----------|--------|----------|
| Domain Classifier | ✅ Done | 95.08% |
| Casting Specialist | ✅ Done | - |
| Steel Specialist | ✅ Done | - |
| 27 Other Specialists | ⏳ Training | - |

## Model

**MSANet (Multi-Scale Attention Network)**
- Backbone: Custom CNN with attention mechanisms
- Input: 224x224 RGB
- Output: Variable classes per domain

## License

MIT
