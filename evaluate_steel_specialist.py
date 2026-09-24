import torch
from torchvision import transforms
from PIL import Image
import os
import config
from models.msa_net import MSANet

checkpoint = torch.load("checkpoints/steel/best_model.pth", map_location="cpu")
class_names = checkpoint["class_names"]
model = MSANet(num_classes=len(class_names), base_channels=config.BASE_CHANNELS)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
])

test_dir = "datasets/steel/test"
y_true, y_pred = [], []

for cls_idx, cls in enumerate(class_names):
    cls_path = os.path.join(test_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    for fname in os.listdir(cls_path):
        img_path = os.path.join(cls_path, fname)
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
        y_true.append(cls_idx)
        y_pred.append(pred)

n = len(class_names)
cm = [[0]*n for _ in range(n)]
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

print()
print("=== OLD STEEL MODEL (MSANet from scratch) ===")
print()
header = "Class".ljust(18) + "Precision".rjust(10) + "Recall".rjust(10) + "F1".rjust(10) + "Support".rjust(10)
print(header)
for i in range(n):
    tp = cm[i][i]
    fp = sum(cm[j][i] for j in range(n) if j != i)
    fn = sum(cm[i][j] for j in range(n) if j != i)
    support = sum(cm[i])
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    row = class_names[i].ljust(18) + f"{precision:10.2f}" + f"{recall:10.2f}" + f"{f1:10.2f}" + f"{support:10d}"
    print(row)

overall_acc = sum(cm[i][i] for i in range(n)) / sum(sum(row) for row in cm)
print()
print(f"Overall Accuracy: {overall_acc*100:.2f}%")
print()
print("Confusion Matrix rows=actual cols=predicted")
for i, row in enumerate(cm):
    print(class_names[i].ljust(18), row)
print()
print("Class index legend:")
for i, c in enumerate(class_names):
    print(f"  {i}: {c}")
