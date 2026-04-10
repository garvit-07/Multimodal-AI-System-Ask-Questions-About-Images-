"""
helpers.py — Utility functions: metrics, checkpointing, logging, visualisation
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# ── Accuracy ─────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy."""
    top_k = logits.topk(k, dim=-1).indices            # (B, k)
    correct = top_k.eq(labels.unsqueeze(-1).expand_as(top_k))
    return correct.any(dim=-1).float().mean().item()


# ── VQA Score (official soft metric) ─────────────────────────────────────
#
# VQAv2 uses soft scoring: if your answer matches n out of 10 human answers,
# you get min(n/3, 1) credit. Here we simplify to exact-match accuracy,
# which is standard for classification training.

def vqa_score(predicted: str, gt_answers: List[str]) -> float:
    """Soft VQA score for a single question."""
    predicted = predicted.strip().lower()
    matches = sum(1 for a in gt_answers if a.strip().lower() == predicted)
    return min(matches / 3.0, 1.0)




def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(path: str, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_acc", 0.0)
    print(f"[Checkpoint] Loaded from {path}  (epoch {epoch}, best_acc {best_acc:.4f})")
    return epoch, best_acc


# ── Answer vocabulary I/O ─────────────────────────────────────────────────

def save_answer_vocab(answer2idx: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(answer2idx, f)
    print(f"[Vocab] Saved answer vocab ({len(answer2idx):,} answers) → {path}")


def load_answer_vocab(path: str) -> Tuple[dict, dict]:
    with open(path) as f:
        answer2idx = json.load(f)
    idx2answer = {int(v): k for k, v in answer2idx.items()}
    print(f"[Vocab] Loaded answer vocab: {len(answer2idx):,} answers")
    return answer2idx, idx2answer


# ── Training curve plotter ────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "checkpoints/training_curves.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train Loss", color="#E85D24")
    axes[0].plot(epochs, val_losses,   label="Val Loss",   color="#3B8BD4")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="Train Acc", color="#E85D24")
    axes[1].plot(epochs, val_accs,   label="Val Acc",   color="#3B8BD4")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


# ── Image de-normalisation for display ────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor back to uint8 numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = std * img + mean
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
