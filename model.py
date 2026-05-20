import torch
import torch.nn as nn
from transformers import AutoModel


class EvidentialDeberta(nn.Module):

    def __init__(
        self,
        model_name="microsoft/mdeberta-v3-base",
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
            nn.Softplus()
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0]

        cls_embedding = self.dropout(cls_embedding)

        evidence = self.classifier(cls_embedding)

        alpha = evidence + 1

        return alpha
