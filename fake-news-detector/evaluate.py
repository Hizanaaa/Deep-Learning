"""
Evaluate a trained Fake News Detection model.
"""

import torch
import pandas as pd
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from transformers import DistilBertTokenizerFast

import config

from models.model import FakeNewsClassifier
from models.trainer import evaluate
from utils.dataset import FakeNewsDataset
from utils.metrics import calculate_metrics


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    config.MODEL_NAME
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)

dataset = FakeNewsDataset(
    test_df["text"].tolist(),
    test_df["label"].tolist(),
    tokenizer,
    config.MAX_LENGTH,
)

loader = DataLoader(
    dataset,
    batch_size=config.VALID_BATCH_SIZE,
)

model = FakeNewsClassifier(
    config.MODEL_NAME
)

model.load_state_dict(
    torch.load(
        f"{config.CHECKPOINT_DIR}/{config.MODEL_SAVE_NAME}",
        map_location=device,
    )
)

model.to(device)

criterion = nn.CrossEntropyLoss()

loss, predictions, labels = evaluate(
    model,
    loader,
    criterion,
    device,
)

metrics = calculate_metrics(
    labels,
    predictions,
)

print("=" * 60)

print(metrics)

print("=" * 60)

print(
    classification_report(
        labels,
        predictions,
    )
)

print("=" * 60)

print(
    confusion_matrix(
        labels,
        predictions,
    )
)