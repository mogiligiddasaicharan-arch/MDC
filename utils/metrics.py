"""
Metrics helpers: accuracy, per-class precision/recall/F1, confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, class_names):
    """Returns a dict with overall accuracy and per-class precision/recall/F1."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    macro_f1 = float(np.mean(f1))

    return {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def print_classification_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names,
                                  zero_division=0))


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (row-normalized)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                     ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")
