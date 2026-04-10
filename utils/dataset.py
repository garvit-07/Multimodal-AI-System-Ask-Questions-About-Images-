"""
dataset.py — PyTorch Dataset for VQA v2
Official dataset: https://visualqa.org/download.html

Expected directory layout under data_root:
  data_root/
    v2_OpenEnded_mscoco_train2014_questions.json
    v2_OpenEnded_mscoco_val2014_questions.json
    v2_mscoco_train2014_annotations.json
    v2_mscoco_val2014_annotations.json
    images/
      train2014/COCO_train2014_000000XXXXXX.jpg
      val2014/COCO_val2014_000000XXXXXX.jpg
"""

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter
from typing import Tuple, Optional, List


# ── Standard image transforms ────────────────────────────────────────────

def get_transform(split: str = "train") -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ── Answer vocabulary builder ────────────────────────────────────────────

def build_answer_vocab(annotations: List[dict], top_k: int = 3129) -> Tuple[dict, dict]:
    """
    VQAv2 has 10 human answers per question. We pick the most common answer
    as the ground-truth label and build a vocabulary of the top_k answers.
    Returns: (answer2idx, idx2answer)
    """
    counter = Counter()
    for ann in annotations:
        for ans in ann["answers"]:
            counter[ans["answer"].strip().lower()] += 1

    top_answers = [a for a, _ in counter.most_common(top_k)]
    answer2idx = {ans: idx for idx, ans in enumerate(top_answers)}
    idx2answer = {idx: ans for ans, idx in answer2idx.items()}
    return answer2idx, idx2answer


# ── Main Dataset ─────────────────────────────────────────────────────────

class VQAv2Dataset(Dataset):
    """
    PyTorch Dataset for VQA v2.

    Args:
        data_root    : path to the data/ folder
        split        : "train" or "val"
        tokenizer    : VQATokenizer instance (already built/loaded)
        answer2idx   : answer vocabulary dict (built once, shared between splits)
        max_samples  : cap the dataset size (useful for debugging)
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        tokenizer,
        answer2idx: dict,
        max_samples: Optional[int] = None,
    ):
        self.data_root  = data_root
        self.split      = split
        self.tokenizer  = tokenizer
        self.answer2idx = answer2idx
        self.transform  = get_transform(split)
        self.num_answers = len(answer2idx)

        # ── Load questions ────────────────────────────────────────────────
        q_file = os.path.join(
            data_root,
            f"v2_OpenEnded_mscoco_{split}2014_questions.json",
        )
        with open(q_file) as f:
            q_data = json.load(f)

        # ── Load annotations ──────────────────────────────────────────────
        ann_file = os.path.join(
            data_root,
            f"v2_mscoco_{split}2014_annotations.json",
        )
        with open(ann_file) as f:
            ann_data = json.load(f)

        # Index annotations by question_id for fast lookup
        ann_by_qid = {a["question_id"]: a for a in ann_data["annotations"]}

        # ── Build sample list ─────────────────────────────────────────────
        self.samples = []
        for q in q_data["questions"]:
            qid = q["question_id"]
            if qid not in ann_by_qid:
                continue

            ann = ann_by_qid[qid]
            # Most common human answer
            answer_counts = Counter(a["answer"].strip().lower() for a in ann["answers"])
            best_answer   = answer_counts.most_common(1)[0][0]

            if best_answer not in self.answer2idx:
                continue  # skip rare answers not in our vocab

            img_id = str(q["image_id"]).zfill(12)
            img_path = os.path.join(
                data_root, "images", f"{split}2014",
                f"COCO_{split}2014_{img_id}.jpg",
            )

            self.samples.append({
                "question_id" : qid,
                "image_path"  : img_path,
                "question"    : q["question"],
                "answer"      : best_answer,
                "answer_idx"  : self.answer2idx[best_answer],
            })

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"[Dataset] {split} set: {len(self.samples):,} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        sample = self.samples[idx]

        # Image
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except FileNotFoundError:
            # Fallback: return a black image so training doesn't crash during testing
            img = Image.new("RGB", (224, 224))
        img = self.transform(img)

        # Question tokens
        q_ids = torch.tensor(
            self.tokenizer.encode(sample["question"]), dtype=torch.long
        )

        return img, q_ids, sample["answer_idx"], sample["question"]
