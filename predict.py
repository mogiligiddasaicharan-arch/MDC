"""Run inference on a single image using a trained specialist model.

Usage:
    python predict.py --domain steel --image path/to/image.jpg
    python predict.py --domain steel --image path/to/image.jpg --no-gradcam
"""

import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image

import config
from models.msa_net import MSANet
from data.dataset import get_transforms
from utils.gradcam import GradCAM, overlay_heatmap


def load_model(device, domain):
    ckpt_path = os.path.join("checkpoints", domain, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Run train.py --domain {domain} first."
        )
    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = MSANet(num_classes=len(class_names),
                    base_channels=config.BASE_CHANNELS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(image_path, model, class_names, device, save_gradcam=True):
    pil_img = Image.open(image_path).convert("RGB")
    transform = get_transforms("val")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())

    print(f"\nPrediction: {class_names[pred_idx]}  (confidence: {confidence:.2%})")
    print("\nAll class probabilities:")
    for i, cls in enumerate(class_names):
        print(f"  {cls:20s}: {probs[i].item():.2%}")

    if save_gradcam:
        input_tensor.requires_grad_(False)
        cam_generator = GradCAM(model, target_layer=model.stage4)
        input_tensor_grad = input_tensor.clone()
        heatmap, cam_class_idx = cam_generator.generate(input_tensor_grad, class_idx=pred_idx)

        original_resized = pil_img.resize((config.IMG_SIZE, config.IMG_SIZE))
        original_bgr = cv2.cvtColor(np.array(original_resized), cv2.COLOR_RGB2BGR)
        overlay = overlay_heatmap(original_bgr, heatmap)

        save_path = os.path.join(config.OUTPUT_DIR, "gradcam_result.png")
        cv2.imwrite(save_path, overlay)
        print(f"\nGrad-CAM visualization saved to {save_path}")

    return class_names[pred_idx], confidence


def main():
    parser = argparse.ArgumentParser(description="MSA-Net single-image inference")
    parser.add_argument("--domain", type=str, required=True,
                         help="Domain name (must match checkpoints/<domain>/)")
    parser.add_argument("--image", type=str, required=True,
                         help="Path to the image file to classify")
    parser.add_argument("--no-gradcam", action="store_true",
                         help="Skip Grad-CAM visualization (faster)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = config.DEVICE
    model, class_names = load_model(device, args.domain)
    predict_image(args.image, model, class_names, device,
                   save_gradcam=not args.no_gradcam)


if __name__ == "__main__":
    main()
