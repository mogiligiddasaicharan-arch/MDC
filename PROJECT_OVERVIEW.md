# MANUFACTURING DEFECT CLASSIFICATION - PROJECT OVERVIEW

## What We Built

A **hierarchical deep learning system** for classifying manufacturing defects across 30 different industrial domains.

### Architecture
```
Input Image
    │
    ▼
┌─────────────────────┐
│  Domain Classifier  │  ← 1 model, 30 classes
│  (MSANet backbone)  │     Trained: 95.08% val accuracy
└─────────────────────┘
    │
    ▼
Routes to correct specialist:
┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐
│ Casting │ │  Steel  │ │Concrete │ ... │Welding  │
│Specialist│ │Specialist│ │Specialist│     │Specialist│
│ 2 classes│ │ 6 classes│ │ 2 classes│     │ N classes│
└─────────┘ └─────────┘ └─────────┘     └─────────┘
```

### Components

| Component | Status | Details |
|-----------|--------|---------|
| **Domain Classifier** | ✅ DONE | 30-class routing model, 95.08% accuracy |
| **Casting Specialist** | ✅ DONE | Defect/no-defect classification |
| **Steel Specialist** | ✅ DONE | 6-class steel defect classification |
| **Concrete Specialist** | ⏳ PENDING | 2-class (crack/no_crack) |
| **Fabric Specialist** | ⏳ PENDING | Fabric defect detection |
| **GC10 Specialist** | ⏳ PENDING | GC-10 dataset defects |
| **Leather Specialist** | ⏳ PENDING | Leather quality inspection |
| **Magnetic Specialist** | ⏳ PENDING | Magnetic tile defects |
| **MVTec Specialists** | ⏳ PENDING | 15 MVTec AD objects (bottle, cable, capsule, etc.) |
| **PCB Specialists** | ⏳ PENDING | pcb, pcb2, pcb_aoi |
| **Severstal Specialist** | ⏳ PENDING | Steel defect segmentation |
| **Solar Specialists** | ⏳ PENDING | solar_elpv, solar_panel |
| **Solder Specialist** | ⏳ PENDING | Solder joint defects |
| **Welding Specialist** | ⏳ PENDING | Welding defect classification |

**Total: 3/30 trained, 27 remaining**

### Model Architecture: MSANet
- Multi-Scale Attention Network
- Base channels: configurable
- Input: 224x224 RGB images
- Output: N classes (domain-dependent)

### Training Details
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early Stopping: patience=5-7
- Batch Size: 32
- Epochs: 25-40
- Device: CPU (no GPU available)

---

## Project Structure

```
manufacturing_defect_classification/
│
├── config.py                    # Global configuration
├── models/
│   ├── msa_net.py              # MSANet architecture
│   └── domain_classifier.py    # Domain classifier model
│
├── datasets/                    # Training data
│   ├── domain_classifier/      # 30-class training data
│   ├── casting/                # Casting defect images
│   ├── steel/                  # Steel defect images
│   ├── concrete/               # Concrete crack images
│   ├── fabric/                 # Fabric defect images
│   ├── gc10/                   # GC-10 defect dataset
│   ├── leather/                # Leather quality images
│   ├── magnetic/               # Magnetic tile defects
│   ├── mvtec_bottle/           # MVTec bottle anomalies
│   ├── mvtec_cable/            # MVTec cable anomalies
│   ├── mvtec_capsule/          # MVTec capsule anomalies
│   ├── mvtec_carpet/           # MVTec carpet anomalies
│   ├── mvtec_grid/             # MVTec grid anomalies
│   ├── mvtec_hazelnut/         # MVTec hazelnut anomalies
│   ├── mvtec_leather/          # MVTec leather anomalies
│   ├── mvtec_metal_nut/        # MVTec metal nut anomalies
│   ├── mvtec_pill/             # MVTec pill anomalies
│   ├── mvtec_screw/            # MVTec screw anomalies
│   ├── mvtec_tile/             # MVTec tile anomalies
│   ├── mvtec_toothbrush/       # MVTec toothbrush anomalies
│   ├── mvtec_transistor/       # MVTec transistor anomalies
│   ├── mvtec_wood/             # MVTec wood anomalies
│   ├── mvtec_zipper/           # MVTec zipper anomalies
│   ├── pcb/                    # PCB defect images
│   ├── pcb2/                   # PCB defect variant 2
│   ├── pcb_aoi/                # PCB AOI inspection
│   ├── severstal/              # Severstal steel defects
│   ├── solar_elpv/             # Solar EL-PV defects
│   ├── solar_panel/            # Solar panel defects
│   ├── solder/                 # Solder joint defects
│   └── welding/                # Welding defect images
│
├── checkpoints/                # Saved model weights
│   ├── domain_classifier/
│   │   └── best_model.pth
│   ├── casting/
│   │   └── best_model.pth
│   ├── steel/
│   │   └── best_model.pth
│   └── [pending domains...]
│
├── train_domain_classifier.py  # Train domain classifier
├── train_all_specialists.py    # Train all specialists
├── train_batch.py              # Smart batch training
├── estimate_time.py            # Time estimation
├── test_pipeline.py            # Full pipeline testing
├── check_gpu.py                # GPU availability check
├── project_status.py           # Project status overview
└── setup_files.py              # File setup helper
```

---

## Training Progress

### Completed
1. ✅ Domain Classifier: 30 classes, 95.08% val accuracy
2. ✅ Casting Specialist: 2 classes
3. ✅ Steel Specialist: 6 classes

### Remaining (27 domains)
| Batch | Domains | Est. Time | When |
|-------|---------|-----------|------|
| Quick Wins | mvtec_toothbrush, mvtec_transistor, mvtec_bottle, mvtec_wood, mvtec_metal_nut | ~3 hours | Today |
| Batch 1 | mvtec_grid, mvtec_tile, mvtec_capsule, mvtec_cable, mvtec_leather, mvtec_zipper, mvtec_carpet, mvtec_pill, solder, mvtec_screw | ~8 hours | Tonight |
| Batch 2 | mvtec_hazelnut, pcb, solar_panel, welding, pcb_aoi, magnetic | ~14 hours | Next night |
| Batch 3 | gc10, solar_elpv, fabric, leather | ~23 hours | Next night |
| Batch 4 | pcb2, concrete, severstal | ~107 hours | Long run |

**Total estimated: ~155 hours (~6.5 days) on CPU**

---

## How to Test What We Built

### 1. Test Domain Classifier
```bash
python test_pipeline.py --mode domain --image path/to/image.jpg
```

### 2. Test Full Pipeline (Domain + Specialist)
```bash
python test_pipeline.py --mode full --image path/to/image.jpg
```

### 3. Test Specific Specialist
```bash
python test_pipeline.py --mode specialist --domain steel --image path/to/image.jpg
```

### 4. Batch Test Directory
```bash
python test_pipeline.py --mode batch --dir path/to/test_images/ --output results.csv
```

---

## GitHub Repository Setup

### Initial Push
```bash
# 1. Initialize repository
git init

# 2. Add all files
git add .

# 3. Commit
git commit -m "Initial commit: Manufacturing Defect Classification"

# 4. Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/manufacturing-defect-classification.git

# 5. Push
git push -u origin main
```

### What to Include/Exclude

**Include:**
- All `.py` source files
- `config.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

**Exclude (add to .gitignore):**
- `datasets/` (too large, use DVC or Git LFS)
- `checkpoints/` (model weights, use releases)
- `__pycache__/` (Python cache)
- `*.pyc` (compiled Python)
- `.idea/`, `.vscode/` (IDE files)

---

## Next Steps

1. **Finish Training**: Use `train_batch.py` to complete all 27 specialists
2. **Create Test Pipeline**: Build `test_pipeline.py` for inference
3. **GitHub Push**: Version control the codebase
4. **Documentation**: Write comprehensive README
5. **Deployment**: Create API or web interface
6. **Evaluation**: Generate confusion matrices and metrics for all domains
