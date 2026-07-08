"""
FastAPI inference server.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from fastapi import FastAPI

from pydantic import BaseModel

from transformers import DistilBertTokenizerFast

import config

from models.model import FakeNewsClassifier


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

app = FastAPI(
    title="Fake News Detection API",
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


class News(BaseModel):

    text: str


@app.get("/")
def root():

    return {
        "message": "Fake News Detection API Running"
    }


@app.post("/predict")
def predict(news: News):

    encoding = tokenizer(
        news.text,
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():

        outputs = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        )

        prediction = torch.argmax(
            outputs,
            dim=1,
        ).item()

        probability = torch.softmax(
            outputs,
            dim=1,
        )[0][prediction].item()

    label = "REAL"

    if prediction == 1:
        label = "FAKE"

    return {
        "prediction": label,
        "confidence": round(probability, 4),
    }