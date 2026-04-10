"""
tokenizer.py — Build and use a simple word-level tokenizer for VQA questions
"""

import re
import json
import os
from collections import Counter
from typing import List, Dict


SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
MAX_QUESTION_LEN = 20


def tokenize(text: str) -> List[str]:
    """Lowercase and split on punctuation/spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class VQATokenizer:
    """
    Simple word-level tokenizer.
    Builds vocabulary from training questions, then encodes/decodes sequences.
    """

    def __init__(self, max_len: int = MAX_QUESTION_LEN):
        self.max_len = max_len
        self.word2idx: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.idx2word: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self.vocab_size = len(SPECIAL_TOKENS)

    # ── Build vocabulary ──────────────────────────────────────────────────

    def build_vocab(self, questions: List[str], min_freq: int = 3, max_vocab: int = 10000):
        """
        questions : list of raw question strings
        min_freq  : minimum token occurrence to include
        max_vocab : hard cap on vocabulary size
        """
        counter = Counter()
        for q in questions:
            counter.update(tokenize(q))

        # Keep only frequent words, sorted by frequency
        common = [w for w, c in counter.most_common(max_vocab) if c >= min_freq]

        for word in common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)
        print(f"[Tokenizer] Vocabulary size: {self.vocab_size:,}")

    # ── Encode / Decode ───────────────────────────────────────────────────

    def encode(self, question: str) -> List[int]:
        """Returns a padded list of token ids of length max_len."""
        tokens = tokenize(question)[: self.max_len]
        ids = [self.word2idx.get(t, SPECIAL_TOKENS["<UNK>"]) for t in tokens]
        # Pad or truncate to max_len
        ids += [SPECIAL_TOKENS["<PAD>"]] * (self.max_len - len(ids))
        return ids[: self.max_len]

    def decode(self, ids: List[int]) -> str:
        return " ".join(
            self.idx2word.get(i, "<UNK>")
            for i in ids
            if i not in (SPECIAL_TOKENS["<PAD>"], SPECIAL_TOKENS["<EOS>"])
        )

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "max_len": self.max_len}, f)
        print(f"[Tokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VQATokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(max_len=data["max_len"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(v): k for k, v in data["word2idx"].items()}
        tok.vocab_size = len(tok.word2idx)
        print(f"[Tokenizer] Loaded vocab size: {tok.vocab_size:,}")
        return tok
