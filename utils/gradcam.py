"""
Grad-CAM implementation for visualizing what MSA-Net focuses on when
classifying a defect. Hooks the last conv feature map (stage4 output)
and its gradients to build a class-discriminative heatmap.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: the MSANet instance (already .eval()'d and on correct device).
            target_layer: the nn.Module whose output feature map we visualize
                           (e.g. model.stage4).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Args:
            input_tensor: normalized image tensor, shape [1, C, H, W]
            class_idx: which class to explain; defaults to the predicted class
        Returns:
            heatmap: numpy array [H, W] in range [0, 1]
            predicted_class: int
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        # Global-average-pool the gradients -> channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1,1,H,W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx


def overlay_heatmap(original_img_bgr, heatmap, alpha=0.45):
    """
    Args:
        original_img_bgr: original image as numpy array (H, W, 3), BGR, uint8
        heatmap: 2D array in [0, 1], any resolution (will be resized)
    Returns:
        overlay: numpy array (H, W, 3), BGR, uint8
    """
    h, w = original_img_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap_color, alpha, original_img_bgr, 1 - alpha, 0)
    return overlay
