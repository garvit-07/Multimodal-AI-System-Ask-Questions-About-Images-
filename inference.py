"""
inference.py — Run inference on a single image + question

Usage:
    python inference.py --image sample_images/dog.jpg --question "What animal is this?"
    python inference.py --image my_photo.jpg --question "How many people are in the image?"
"""

import os
import sys
import argparse
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from models.vqa_model import VQAModel
from utils.tokenizer  import VQATokenizer
from utils.dataset    import get_transform
from utils.helpers    import load_answer_vocab


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True,          help="Path to input image")
    p.add_argument("--question",   required=True,          help="Question about the image")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--top_k",      type=int, default=5,    help="Number of top answers to show")
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    tokenizer      = VQATokenizer.load(os.path.join(checkpoint_dir, "tokenizer.json"))
    answer2idx, idx2answer = load_answer_vocab(os.path.join(checkpoint_dir, "answer_vocab.json"))

    ckpt       = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt.get("args", {})

    model = VQAModel(
        vocab_size  = tokenizer.vocab_size,
        num_answers = len(answer2idx),
        embed_dim   = saved_args.get("embed_dim", 512),
        use_bert    = saved_args.get("use_bert", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, idx2answer


def predict(image_path: str, question: str, model, tokenizer, idx2answer, device, top_k=5):
    # Preprocess image
    transform = get_transform("val")
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    # Encode question
    q_ids = torch.tensor([tokenizer.encode(question)], dtype=torch.long).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor, q_ids)
        probs  = torch.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(top_k, dim=-1)

    results = []
    for i in range(top_k):
        ans  = idx2answer[top_ids[0, i].item()]
        conf = top_probs[0, i].item()
        results.append((ans, conf))
    return results


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Inference] Image    : {args.image}")
    print(f"[Inference] Question : {args.question}")
    print(f"[Inference] Device   : {device}\n")

    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        print("  → Train the model first: python train.py --debug")
        sys.exit(1)

    model, tokenizer, idx2answer = load_model(args.checkpoint, device)
    results = predict(args.image, args.question, model, tokenizer, idx2answer, device, args.top_k)

    print(f"{'─'*40}")
    print(f"  Question: {args.question}")
    print(f"{'─'*40}")
    for rank, (ans, conf) in enumerate(results, 1):
        bar = "█" * int(conf * 30)
        print(f"  {rank}. {ans:<20} {conf*100:5.1f}%  {bar}")
    print(f"{'─'*40}")
    print(f"  Top answer: {results[0][0]}")


if __name__ == "__main__":
    main()
