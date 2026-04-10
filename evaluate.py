"""
evaluate.py — Evaluate a trained VQA model on the validation set

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from models.vqa_model import VQAModel
from utils.tokenizer  import VQATokenizer
from utils.dataset    import VQAv2Dataset
from utils.helpers    import accuracy, top_k_accuracy, load_checkpoint, load_answer_vocab


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VQA model")
    p.add_argument("--checkpoint",   required=True,       help="Path to .pth checkpoint")
    p.add_argument("--data_root",    default="data")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--max_samples",  type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, idx2answer, device):
    model.eval()
    total_acc    = 0.0
    total_top5   = 0.0
    n_batches    = 0

    # Per-question-type accuracy
    type_correct = defaultdict(int)
    type_total   = defaultdict(int)

    for images, questions, labels, raw_questions in tqdm(loader, desc="Evaluating"):
        images    = images.to(device)
        questions = questions.to(device)
        labels    = labels.to(device)

        logits = model(images, questions)

        total_acc  += accuracy(logits, labels)
        total_top5 += top_k_accuracy(logits, labels, k=5)
        n_batches  += 1

        # Simple question-type bucketing (yes/no / number / other)
        preds = logits.argmax(dim=-1)
        for i, q in enumerate(raw_questions):
            q_lower = q.lower()
            if q_lower.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ", "have ", "can ")):
                qtype = "yes/no"
            elif q_lower.startswith(("how many", "how much", "what number")):
                qtype = "number"
            else:
                qtype = "other"

            correct = int(preds[i].item() == labels[i].item())
            type_correct[qtype] += correct
            type_total[qtype]   += 1

    print(f"\n{'─'*40}")
    print(f"  Overall Top-1 Accuracy : {total_acc  / n_batches:.4f}")
    print(f"  Overall Top-5 Accuracy : {total_top5 / n_batches:.4f}")
    print(f"{'─'*40}")
    print("  Per question type:")
    for qtype in ["yes/no", "number", "other"]:
        if type_total[qtype]:
            acc = type_correct[qtype] / type_total[qtype]
            print(f"    {qtype:<10}: {acc:.4f}  ({type_total[qtype]:,} samples)")
    print(f"{'─'*40}\n")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    checkpoint_dir = os.path.dirname(args.checkpoint)
    tokenizer   = VQATokenizer.load(os.path.join(checkpoint_dir, "tokenizer.json"))
    answer2idx, idx2answer = load_answer_vocab(os.path.join(checkpoint_dir, "answer_vocab.json"))

    val_ds = VQAv2Dataset(
        args.data_root, "val", tokenizer, answer2idx,
        max_samples=args.max_samples,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    # Load model config from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    model = VQAModel(
        vocab_size  = tokenizer.vocab_size,
        num_answers = len(answer2idx),
        embed_dim   = saved_args.get("embed_dim", 512),
        use_bert    = saved_args.get("use_bert", False),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print(f"[Eval] Loaded checkpoint: {args.checkpoint}")

    evaluate(model, val_loader, idx2answer, device)


if __name__ == "__main__":
    main()
