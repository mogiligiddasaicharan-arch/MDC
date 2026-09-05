"""
Test Pipeline for Manufacturing Defect Classification

Usage:
    python test_pipeline.py --mode domain --image path/to/image.jpg
    python test_pipeline.py --mode full --image path/to/image.jpg
    python test_pipeline.py --mode specialist --domain steel --image path/to/image.jpg
    python test_pipeline.py --mode batch --dir path/to/images/ --output results.csv
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

import config
from models.msa_net import MSANet


# Domain to expected classes mapping (update as you train)
DOMAIN_CLASSES = {
    "domain_classifier": 30,
    "casting": 2,
    "steel": 6,
    "concrete": 2,
    "fabric": None,  # Update after training
    "gc10": None,
    "leather": None,
    "magnetic": None,
    "mvtec_bottle": None,
    "mvtec_cable": None,
    "mvtec_capsule": None,
    "mvtec_carpet": None,
    "mvtec_grid": None,
    "mvtec_hazelnut": None,
    "mvtec_leather": None,
    "mvtec_metal_nut": None,
    "mvtec_pill": None,
    "mvtec_screw": None,
    "mvtec_tile": None,
    "mvtec_toothbrush": None,
    "mvtec_transistor": None,
    "mvtec_wood": None,
    "mvtec_zipper": None,
    "pcb": None,
    "pcb2": None,
    "pcb_aoi": None,
    "severstal": None,
    "solar_elpv": None,
    "solar_panel": None,
    "solder": None,
    "welding": None,
}

# Domain names list (must match training order)
DOMAIN_NAMES = [
    "casting", "concrete", "fabric", "gc10", "leather", "magnetic",
    "mvtec_bottle", "mvtec_cable", "mvtec_capsule", "mvtec_carpet",
    "mvtec_grid", "mvtec_hazelnut", "mvtec_leather", "mvtec_metal_nut",
    "mvtec_pill", "mvtec_screw", "mvtec_tile", "mvtec_toothbrush",
    "mvtec_transistor", "mvtec_wood", "mvtec_zipper", "pcb", "pcb2",
    "pcb_aoi", "severstal", "solar_elpv", "solar_panel", "solder",
    "steel", "welding"
]


def load_model(checkpoint_path, num_classes, device):
    """Load a trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        return None

    model = MSANet(num_classes=num_classes, base_channels=config.BASE_CHANNELS).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("class_names", [f"class_{i}" for i in range(num_classes)])


def preprocess_image(image_path):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_domain(image_tensor, device):
    """Predict which domain an image belongs to."""
    ckpt_path = os.path.join("checkpoints", "domain_classifier", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print("ERROR: Domain classifier not found!")
        return None, None

    model, class_names = load_model(ckpt_path, 30, device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    domain_idx = predicted.item()
    domain_name = class_names[domain_idx] if domain_idx < len(class_names) else DOMAIN_NAMES[domain_idx]
    return domain_name, confidence.item()


def predict_defect(image_tensor, domain, device):
    """Predict defect class within a specific domain."""
    ckpt_path = os.path.join("checkpoints", domain, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Specialist for '{domain}' not found!")
        return None, None, None

    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names)

    model, _ = load_model(ckpt_path, num_classes, device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    defect_class = class_names[predicted.item()] if predicted.item() < len(class_names) else f"class_{predicted.item()}"
    all_probs = {class_names[i]: probs[0][i].item() for i in range(len(class_names))}

    return defect_class, confidence.item(), all_probs


def test_single_image(image_path, mode="full", domain=None):
    """Test a single image through the pipeline."""
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*60}")

    try:
        image_tensor = preprocess_image(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    if mode == "domain":
        domain_name, conf = predict_domain(image_tensor, device)
        print(f"Predicted Domain: {domain_name}")
        print(f"Confidence: {conf:.4f}")
        return {"domain": domain_name, "confidence": conf}

    elif mode == "specialist":
        if domain is None:
            print("ERROR: --domain required for specialist mode")
            return None
        defect, conf, probs = predict_defect(image_tensor, domain, device)
        print(f"Domain: {domain}")
        print(f"Predicted Defect: {defect}")
        print(f"Confidence: {conf:.4f}")
        print(f"All Probabilities: {probs}")
        return {"domain": domain, "defect": defect, "confidence": conf, "all_probs": probs}

    elif mode == "full":
        # Step 1: Domain classification
        domain_name, domain_conf = predict_domain(image_tensor, device)
        print(f"Step 1 - Domain: {domain_name} (confidence: {domain_conf:.4f})")

        # Step 2: Specialist prediction
        defect, defect_conf, probs = predict_defect(image_tensor, domain_name, device)
        if defect:
            print(f"Step 2 - Defect: {defect} (confidence: {defect_conf:.4f})")
            print(f"All probabilities: {probs}")

        return {
            "domain": domain_name,
            "domain_confidence": domain_conf,
            "defect": defect,
            "defect_confidence": defect_conf,
            "all_probs": probs
        }


def test_batch(image_dir, output_file="results.csv"):
    """Test all images in a directory."""
    device = config.DEVICE
    results = []

    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

    print(f"\nTesting {len(image_files)} images from {image_dir}")

    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        try:
            image_tensor = preprocess_image(img_path)
            domain_name, domain_conf = predict_domain(image_tensor, device)
            defect, defect_conf, probs = predict_defect(image_tensor, domain_name, device)

            result = {
                "filename": img_file,
                "predicted_domain": domain_name,
                "domain_confidence": domain_conf,
                "predicted_defect": defect,
                "defect_confidence": defect_conf,
            }

            # Add individual class probabilities
            if probs:
                for cls, prob in probs.items():
                    result[f"prob_{cls}"] = prob

            results.append(result)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            results.append({
                "filename": img_file,
                "error": str(e)
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary:")
    print(df[["filename", "predicted_domain", "predicted_defect"]].to_string())

    return df


def main():
    parser = argparse.ArgumentParser(description="Test Manufacturing Defect Classification Pipeline")
    parser.add_argument("--mode", choices=["domain", "specialist", "full", "batch"], 
                      default="full", help="Test mode")
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--domain", type=str, help="Domain for specialist mode")
    parser.add_argument("--dir", type=str, help="Directory for batch testing")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV for batch mode")

    args = parser.parse_args()

    if args.mode in ["domain", "specialist", "full"]:
        if not args.image:
            print("ERROR: --image required for single image testing")
            return
        test_single_image(args.image, args.mode, args.domain)

    elif args.mode == "batch":
        if not args.dir:
            print("ERROR: --dir required for batch testing")
            return
        test_batch(args.dir, args.output)


if __name__ == "__main__":
    main()
