"""
Run inference on a single image using a trained MSA-Net checkpoint,
and save a Grad-CAM heatmap overlay showing what the model focused on.

Usage:
    python predict.py --image path/to/image.jpg
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


def load_model(device):
    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {config.BEST_MODEL_PATH}. Run train.py first."
        )
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = MSANet(num_classes=len(class_names),
                    base_channels=config.BASE_CHANNELS).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(image_path, model, class_names, device, save_gradcam=True):
    pil_img = Image.open(image_path).convert("RGB")
    transform = get_transforms("val")   # no augmentation, just resize+normalize
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
        # Need gradients for Grad-CAM, so re-run with grad enabled
        input_tensor.requires_grad_(False)
        cam_generator = GradCAM(model, target_layer=model.stage4)
        input_tensor_grad = input_tensor.clone()
        heatmap, cam_class_idx = cam_generator.generate(input_tensor_grad, class_idx=pred_idx)

        # Prepare original image for overlay (resize to model input size, BGR for cv2)
        original_resized = pil_img.resize((config.IMG_SIZE, config.IMG_SIZE))
        original_bgr = cv2.cvtColor(np.array(original_resized), cv2.COLOR_RGB2BGR)

        overlay = overlay_heatmap(original_bgr, heatmap)

        save_path = os.path.join(config.OUTPUT_DIR, "gradcam_result.png")
        cv2.imwrite(save_path, overlay)
        print(f"\nGrad-CAM visualization saved to {save_path}")

    return class_names[pred_idx], confidence


def main():
    parser = argparse.ArgumentParser(description="MSA-Net single-image inference")
    parser.add_argument("--image", type=str, required=True,
                         help="Path to the image file to classify")
    parser.add_argument("--no-gradcam", action="store_true",
                         help="Skip Grad-CAM visualization (faster)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    device = config.DEVICE
    model, class_names = load_model(device)
    predict_image(args.image, model, class_names, device,
                   save_gradcam=not args.no_gradcam)


if __name__ == "__main__":
    main()
