"""
Project configuration.
"""

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256

TRAIN_BATCH_SIZE = 16

VALID_BATCH_SIZE = 16

EPOCHS = 5

LEARNING_RATE = 2e-5

RANDOM_SEED = 42

CHECKPOINT_DIR = "checkpoints"

MODEL_SAVE_NAME = "bert_fake_news.pt"