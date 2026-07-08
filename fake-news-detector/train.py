"""
Training entry point.
"""

import os

import pandas as pd

import torch

from torch.utils.data import DataLoader

from transformers import (
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

import torch.nn as nn

from models.model import FakeNewsClassifier

from models.trainer import (
    train_epoch,
    evaluate,
)

from utils.dataset import FakeNewsDataset

from utils.metrics import calculate_metrics

from utils.seed import set_seed

import config


set_seed(config.RANDOM_SEED)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


train_df = pd.read_csv(
    "data/processed/train.csv"
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)


tokenizer = DistilBertTokenizerFast.from_pretrained(
    config.MODEL_NAME
)


train_dataset = FakeNewsDataset(
    train_df["text"].tolist(),
    train_df["label"].tolist(),
    tokenizer,
    config.MAX_LENGTH,
)

test_dataset = FakeNewsDataset(
    test_df["text"].tolist(),
    test_df["label"].tolist(),
    tokenizer,
    config.MAX_LENGTH,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=config.TRAIN_BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.VALID_BATCH_SIZE,
)


model = FakeNewsClassifier(
    config.MODEL_NAME
)

model.to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
)


total_steps = (
    len(train_loader)
    * config.EPOCHS
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)


criterion = nn.CrossEntropyLoss()


best_accuracy = 0


os.makedirs(
    config.CHECKPOINT_DIR,
    exist_ok=True,
)


for epoch in range(config.EPOCHS):

    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        device,
    )

    val_loss, preds, labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    metrics = calculate_metrics(
        labels,
        preds,
    )

    print("=" * 50)

    print(f"Epoch {epoch + 1}")

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Validation Loss : {val_loss:.4f}")

    print(metrics)

    if metrics["accuracy"] > best_accuracy:

        best_accuracy = metrics["accuracy"]

        torch.save(
            model.state_dict(),
            os.path.join(
                config.CHECKPOINT_DIR,
                config.MODEL_SAVE_NAME,
            ),
        )

        print("Best model saved.")