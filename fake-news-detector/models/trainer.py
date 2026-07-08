"""
Training utilities.
"""

import torch
from tqdm import tqdm


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
):

    model.train()

    running_loss = 0

    for batch in tqdm(dataloader):

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        scheduler.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0

    predictions = []

    targets = []

    for batch in dataloader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(outputs, labels)

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())

        targets.extend(labels.cpu().numpy())

    return (
        running_loss / len(dataloader),
        predictions,
        targets,
    )