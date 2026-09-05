# estimate_time.py - Estimate total training time on CPU
import os

def count_images(domain_path):
    count = 0
    for root, dirs, files in os.walk(domain_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                count += 1
    return count

data_dir = "datasets"
if not os.path.exists(data_dir):
    print("Data directory not found!")
    exit(1)

pending = []
for item in os.listdir(data_dir):
    item_path = os.path.join(data_dir, item)
    if os.path.isdir(item_path) and item != "domain_classifier":
        ckpt = os.path.join("checkpoints", item, "best_model.pth")
        if not os.path.exists(ckpt):
            count = count_images(item_path)
            pending.append((item, count))

pending.sort(key=lambda x: x[1])

print("=" * 60)
print("TRAINING TIME ESTIMATE (CPU, 25 epochs, batch_size=32)")
print("=" * 60)
print(f"{'Domain':<25} {'Images':>8} {'Est. Time':>12}")
print("-" * 60)

total_images = 0
total_time_min = 0
# Rough estimate: ~0.5 seconds per image per epoch on CPU
# With early stopping, average ~15 epochs
SEC_PER_IMG_PER_EPOCH = 0.5
AVG_EPOCHS = 15

for domain, count in pending:
    est_time = (count * SEC_PER_IMG_PER_EPOCH * AVG_EPOCHS) / 60  # minutes
    total_images += count
    total_time_min += est_time
    print(f"{domain:<25} {count:>8} {est_time:>10.0f}m")

print("-" * 60)
print(f"{'TOTAL':<25} {total_images:>8} {total_time_min:>10.0f}m ({total_time_min/60:.1f}h)")
print("=" * 60)
print(f"\nNOTE: Estimates assume ~{SEC_PER_IMG_PER_EPOCH}s/img/epoch on CPU")
print(f"      Early stopping typically kicks in at ~{AVG_EPOCHS} epochs")
print(f"      Actual time may vary by ±30% depending on your CPU")
