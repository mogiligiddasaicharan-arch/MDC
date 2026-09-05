# Manufacturing Defect Classification using MSA-Net

Complete implementation: data loading, MSA-Net model (Multi-Scale Attention
Network), training, evaluation, and single-image inference.

## Project Structure
```
manufacturing_defect_classification/
├── data/
│   └── dataset.py          # Dataset loader + augmentation
├── models/
│   └── msa_net.py           # MSA-Net architecture
├── utils/
│   ├── metrics.py            # Accuracy / F1 / confusion matrix helpers
│   └── gradcam.py            # Grad-CAM visualization
├── train.py                  # Training loop
├── evaluate.py                # Evaluation on test set
├── predict.py                 # Single-image inference
├── config.py                   # All hyperparameters in one place
├── requirements.txt
└── checkpoints/                # Saved model weights (created at runtime)
```

## 1. Setup
```bash
pip install -r requirements.txt
```

## 2. Prepare data
Download the NEU steel surface defect dataset (see earlier instructions —
Kaggle: `fantacher/neu-metal-surface-defects-data`), unzip it, and arrange as:

```
dataset/
├── train/
│   ├── Crazing/
│   ├── Inclusion/
│   ├── Patches/
│   ├── Pitted/
│   ├── Rolled/
│   └── Scratches/
├── val/
│   └── ... same 6 classes
└── test/
    └── ... same 6 classes
```
If your extracted folder uses different names (e.g. `valid` instead of `val`),
either rename the folder or edit `DATA_DIR` / split names in `config.py`.

## 3. Train
```bash
python train.py
```
Trains MSA-Net, saves best weights to `checkpoints/best_model.pth`, and logs
training/validation curves to `outputs/training_curves.png`.

## 4. Evaluate
```bash
python evaluate.py
```
Prints accuracy/precision/recall/F1 per class and saves a confusion matrix
image to `outputs/confusion_matrix.png`.

## 5. Predict on a single image
```bash
python predict.py --image path/to/image.jpg
```
Prints the predicted class + confidence, and saves a Grad-CAM heatmap overlay
to `outputs/gradcam_result.png`.

## Model: MSA-Net
- **Multi-scale feature extraction**: 3 parallel convolution branches per
  block using kernel sizes 3x3, 5x5, and a dilated 3x3 (effective 5x5 receptive
  field), concatenated and fused with a 1x1 conv.
- **Channel attention**: Squeeze-and-Excitation (SE) block.
- **Spatial attention**: CBAM-style spatial attention (max+avg pooling ->
  7x7 conv -> sigmoid mask).
- **Classifier head**: Global Average Pooling -> FC -> Softmax.
