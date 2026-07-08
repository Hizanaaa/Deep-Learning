"""
Predict Fake or Real news from user input.
"""

import torch

from transformers import DistilBertTokenizerFast

import config

from models.model import FakeNewsClassifier


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


tokenizer = DistilBertTokenizerFast.from_pretrained(
    config.MODEL_NAME
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

model.eval()


LABELS = {
    0: "REAL",
    1: "FAKE",
}


while True:

    text = input("\nEnter news text (or 'quit'): ")

    if text.lower() == "quit":
        break

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():

        output = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        )

        prediction = torch.argmax(
            output,
            dim=1,
        ).item()

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence = probabilities[0][prediction].item()

    print()

    print("Prediction :", LABELS[prediction])

    print(f"Confidence : {confidence:.4f}")