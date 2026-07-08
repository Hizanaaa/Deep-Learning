"""
DistilBERT classifier for Fake News Detection.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel


class FakeNewsClassifier(nn.Module):
    """
    DistilBERT + Dropout + Linear layer.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs.last_hidden_state[:, 0]

        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        return logits