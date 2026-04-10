"""
vqa_model.py — Full VQA model: Image + Question → Answer
"""

import torch
import torch.nn as nn
from models.encoders import ImageEncoder, QuestionEncoderLSTM, QuestionEncoderBERT


class FusionModule(nn.Module):
    """
    Fuses visual and textual feature vectors.
    Method: element-wise multiplication (Hadamard) after projection.
    This is more expressive than simple concatenation.
    """

    def __init__(self, embed_dim: int = 512, out_dim: int = 1024):
        super().__init__()
        self.img_proj = nn.Linear(embed_dim, out_dim)
        self.txt_proj = nn.Linear(embed_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        v = torch.relu(self.img_proj(img_feat))  # (B, out_dim)
        q = torch.relu(self.txt_proj(txt_feat))  # (B, out_dim)
        fused = v * q                             # Hadamard product
        return self.norm(fused)                   # (B, out_dim)


class VQAModel(nn.Module):
    """
    Full VQA pipeline:
        Image  → ImageEncoder  → img_feat (B, embed_dim)
        Question → QuestionEncoder → txt_feat (B, embed_dim)
        [img_feat, txt_feat] → FusionModule → MLP → logits (B, num_answers)

    Args:
        vocab_size   : number of tokens in the question vocabulary
        num_answers  : number of answer classes (VQAv2 top-3129 answers)
        embed_dim    : shared embedding dimension (default 512)
        use_bert     : use BERT encoder instead of LSTM (needs transformers)
        freeze_cnn   : freeze ResNet backbone weights
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        num_answers: int = 3129,
        embed_dim: int = 512,
        use_bert: bool = False,
        freeze_cnn: bool = True,
    ):
        super().__init__()
        self.use_bert = use_bert

        # Vision encoder
        self.img_encoder = ImageEncoder(embed_dim=embed_dim, freeze_backbone=freeze_cnn)

        # Text encoder
        if use_bert:
            self.txt_encoder = QuestionEncoderBERT(embed_dim=embed_dim)
        else:
            self.txt_encoder = QuestionEncoderLSTM(vocab_size=vocab_size, embed_dim=embed_dim)

        # Fusion
        self.fusion = FusionModule(embed_dim=embed_dim, out_dim=1024)

        # Answer classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_answers),
        )

    def forward(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        images:         (B, 3, 224, 224)
        questions:      (B, seq_len)         token ids
        attention_mask: (B, seq_len)         required only for BERT
        returns:        (B, num_answers)     raw logits
        """
        img_feat = self.img_encoder(images)

        if self.use_bert:
            txt_feat = self.txt_encoder(questions, attention_mask)
        else:
            txt_feat = self.txt_encoder(questions)

        fused = self.fusion(img_feat, txt_feat)
        return self.classifier(fused)

    def predict(self, images, questions, answer_vocab, attention_mask=None, top_k=3):
        """
        Convenience method: returns top-k (answer_string, confidence) pairs.
        answer_vocab: list of answer strings indexed by class id
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, questions, attention_mask)
            probs  = torch.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(top_k, dim=-1)

        results = []
        for i in range(top_k):
            ans = answer_vocab[top_ids[0, i].item()]
            conf = top_probs[0, i].item()
            results.append((ans, conf))
        return results
