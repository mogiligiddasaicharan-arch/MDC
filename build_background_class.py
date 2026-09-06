import torchvision
import os
from PIL import Image

os.makedirs("datasets/_background/train/other", exist_ok=True)
os.makedirs("datasets/_background/val/other", exist_ok=True)

print("Downloading CIFAR-100 as a generic 'not manufacturing' background class...")
dataset = torchvision.datasets.CIFAR100(root="./_cifar_temp", train=True, download=True)

count = 0
for i in range(0, 2000, 2):
    img, _ = dataset[i]
    img = img.resize((224, 224))
    folder = "train" if count % 5 != 0 else "val"
    img.save(f"datasets/_background/{folder}/other/bg_{count}.png")
    count += 1

print(f"Saved {count} generic background images to datasets/_background/")
