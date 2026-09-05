"""Two-stage inference pipeline: Domain Classifier -> Specialist Model."""

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

import config
from models.msa_net import MSANet
from utils.gradcam import GradCAM, overlay_heatmap


class InferencePipeline:
    def __init__(self, checkpoint_dir="checkpoints", device=None):
        self.device = device or config.DEVICE
        self.checkpoint_dir = checkpoint_dir

        self.domain_classifier, self.domain_classes = self._load_model(
            os.path.join(checkpoint_dir, "domain_classifier", "best_model.pth")
        )

        self.specialist_models = {}
        self.specialist_classes = {}

        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path) and item != "domain_classifier":
                model_path = os.path.join(item_path, "best_model.pth")
                if os.path.exists(model_path):
                    model, classes = self._load_model(model_path)
                    self.specialist_models[item] = model
                    self.specialist_classes[item] = classes
                    print(f"  Loaded specialist: {item} ({len(classes)} classes)")

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ])

    def _load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        class_names = checkpoint["class_names"]
        num_classes = len(class_names)
        model = MSANet(num_classes=num_classes, base_channels=config.BASE_CHANNELS).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, class_names

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor, img

    def predict(self, image_path, save_gradcam_path=None):
        tensor, img = self.preprocess(image_path)

        with torch.no_grad():
            domain_logits = self.domain_classifier(tensor)
            domain_probs = F.softmax(domain_logits, dim=1)[0]

        domain_idx = domain_probs.argmax().item()
        predicted_domain = self.domain_classes[domain_idx]
        domain_confidence = domain_probs[domain_idx].item()

        domain_prob_dict = {cls: domain_probs[i].item() for i, cls in enumerate(self.domain_classes)}

        specialist = self.specialist_models.get(predicted_domain)
        specialist_classes = self.specialist_classes.get(predicted_domain, [])

        if specialist is None:
            return {
                "domain": predicted_domain,
                "domain_confidence": domain_confidence,
                "defect": "UNKNOWN (no specialist model)",
                "defect_confidence": 0.0,
                "domain_probabilities": domain_prob_dict,
                "defect_probabilities": {},
                "gradcam_path": None,
            }

        with torch.no_grad():
            defect_logits = specialist(tensor)
            defect_probs = F.softmax(defect_logits, dim=1)[0]

        defect_idx = defect_probs.argmax().item()
        predicted_defect = specialist_classes[defect_idx]
        defect_confidence = defect_probs[defect_idx].item()

        defect_prob_dict = {cls: defect_probs[i].item() for i, cls in enumerate(specialist_classes)}

        gradcam_path = None
        if save_gradcam_path:
            try:
                gradcam = GradCAM(model=specialist, target_layer=specialist.stage4)
                heatmap, _ = gradcam.generate(tensor, class_idx=defect_idx)
                img_np = np.array(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                overlay = overlay_heatmap(img_bgr, heatmap, alpha=0.45)
                cv2.imwrite(save_gradcam_path, overlay)
                gradcam_path = save_gradcam_path
            except Exception as e:
                print(f"Grad-CAM failed: {e}")

        return {
            "domain": predicted_domain,
            "domain_confidence": domain_confidence,
            "defect": predicted_defect,
            "defect_confidence": defect_confidence,
            "domain_probabilities": domain_prob_dict,
            "defect_probabilities": defect_prob_dict,
            "gradcam_path": gradcam_path,
        }
