"""
train.py — Training script for the VQA model

Usage (quick debug run, no GPU needed):
    python train.py --debug --epochs 2 --batch_size 8 --max_samples 200

Usage (full training):
    python train.py --epochs 30 --batch_size 64 --lr 1e-4
"""

import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from models.vqa_model import VQAModel
from utils.tokenizer  import VQATokenizer
from utils.dataset    import VQAv2Dataset, build_answer_vocab
from utils.helpers    import (
    accuracy, save_checkpoint, load_checkpoint,
    save_answer_vocab, load_answer_vocab,
    plot_training_curves,
)


# ── CLI Arguments ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train VQA model")
    p.add_argument("--data_root",    default="data",          help="Path to data folder")
    p.add_argument("--checkpoint_dir", default="checkpoints", help="Where to save checkpoints")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--embed_dim",    type=int,   default=512)
    p.add_argument("--num_answers",  type=int,   default=3129,  help="Top-N answer classes")
    p.add_argument("--max_vocab",    type=int,   default=10000, help="Question vocab size")
    p.add_argument("--max_samples",  type=int,   default=None,  help="Cap dataset size")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--use_bert",     action="store_true",       help="Use BERT text encoder")
    p.add_argument("--resume",       default=None,              help="Resume from checkpoint path")
    p.add_argument("--debug",        action="store_true",       help="Quick run: 200 samples, 2 epochs")
    return p.parse_args()


# ── Training epoch ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for images, questions, labels, _ in pbar:
        images    = images.to(device)
        questions = questions.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, questions)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        acc = accuracy(logits.detach(), labels)
        total_loss += loss.item()
        total_acc  += acc
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

    n = len(loader)
    return total_loss / n, total_acc / n


# ── Validation epoch ──────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)

    for images, questions, labels, _ in pbar:
        images    = images.to(device)
        questions = questions.to(device)
        labels    = labels.to(device)

        logits = model(images, questions)
        loss   = criterion(logits, labels)
        acc    = accuracy(logits, labels)

        total_loss += loss.item()
        total_acc  += acc

    n = len(loader)
    return total_loss / n, total_acc / n


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.debug:
        print("[Debug mode] Using 200 samples, 2 epochs")
        args.max_samples = 200
        args.epochs      = 2
        args.batch_size  = 8
        args.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Vocabulary paths ─────────────────────────────────────────────────
    tok_path = os.path.join(args.checkpoint_dir, "tokenizer.json")
    ans_path = os.path.join(args.checkpoint_dir, "answer_vocab.json")

    # ── Build tokenizer and answer vocab ─────────────────────────────────
    print("[Setup] Loading / building vocabularies …")

    if os.path.exists(tok_path) and os.path.exists(ans_path):
        tokenizer = VQATokenizer.load(tok_path)
        answer2idx, idx2answer = load_answer_vocab(ans_path)
    else:
        import json as _json
        q_file  = os.path.join(args.data_root, "v2_OpenEnded_mscoco_train2014_questions.json")
        ann_file = os.path.join(args.data_root, "v2_mscoco_train2014_annotations.json")

        with open(q_file)  as f: q_data   = _json.load(f)
        with open(ann_file) as f: ann_data = _json.load(f)

        # Build question tokenizer
        tokenizer = VQATokenizer()
        tokenizer.build_vocab(
            [q["question"] for q in q_data["questions"]],
            min_freq=3,
            max_vocab=args.max_vocab,
        )
        tokenizer.save(tok_path)

        # Build answer vocab
        answer2idx, idx2answer = build_answer_vocab(
            ann_data["annotations"], top_k=args.num_answers
        )
        save_answer_vocab(answer2idx, ans_path)

    args.num_answers = len(answer2idx)

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    train_ds = VQAv2Dataset(
        args.data_root, "train", tokenizer, answer2idx,
        max_samples=args.max_samples,
    )
    val_ds = VQAv2Dataset(
        args.data_root, "val", tokenizer, answer2idx,
        max_samples=args.max_samples // 5 if args.max_samples else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Model, optimiser, scheduler ──────────────────────────────────────
    model = VQAModel(
        vocab_size  = tokenizer.vocab_size,
        num_answers = args.num_answers,
        embed_dim   = args.embed_dim,
        use_bert    = args.use_bert,
        freeze_cnn  = True,
    ).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_acc = load_checkpoint(args.resume, model, optimizer, device)

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss,   val_acc   = val_epoch  (model, val_loader,               criterion, device, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_val_acc,
                    "args": vars(args),
                },
                os.path.join(args.checkpoint_dir, "best_model.pth"),
            )

        # Save latest checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "best_acc": best_val_acc},
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pth"),
            )

    # ── Final plots ───────────────────────────────────────────────────────
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"],  history["val_acc"],
    )
    print(f"\n[Done] Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
