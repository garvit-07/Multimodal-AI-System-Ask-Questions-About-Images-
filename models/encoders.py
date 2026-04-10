"""
encoders.py — Vision and Language encoders for VQA
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────
#  Vision Encoder  (ResNet-50 backbone)
# ─────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """
    Extracts a fixed-size visual feature vector from an image.
    Backbone: ResNet-50 pretrained on ImageNet (final FC removed).
    Output:   (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 2048, 1, 1)

        # Project to shared embedding space
        self.proj = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224)
        returns: (B, embed_dim)
        """
        feat = self.backbone(x)            # (B, 2048, 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)  # (B, 2048)
        return self.proj(feat)             # (B, embed_dim)


# ─────────────────────────────────────────────
#  Text Encoder  (Embedding + LSTM)
# ─────────────────────────────────────────────

class QuestionEncoderLSTM(nn.Module):
    """
    Encodes a tokenized question into a fixed-size vector.
    Architecture: Embedding → Bi-LSTM → last hidden state projection
    Output: (B, embed_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim // 2,          # bidirectional doubles this → hidden_dim
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, seq_len)  — padded token ids
        returns: (B, embed_dim)
        """
        x = self.embed(tokens)            # (B, seq_len, embed_dim)
        _, (h, _) = self.lstm(x)         # h: (num_layers*2, B, hidden//2)
        # concat last forward + backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden_dim)
        return self.proj(h)               # (B, embed_dim)


# ─────────────────────────────────────────────
#  Text Encoder  (BERT — better accuracy)
# ─────────────────────────────────────────────

class QuestionEncoderBERT(nn.Module):
    """
    Uses pretrained BERT to encode the question.
    Requires: pip install transformers
    Output: (B, embed_dim)
    """

    def __init__(self, embed_dim: int = 512, freeze_bert: bool = False):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids:      (B, seq_len)
        attention_mask: (B, seq_len)
        returns:        (B, embed_dim)
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]  # [CLS] token → (B, 768)
        return self.proj(cls_token)                  # (B, embed_dim)
