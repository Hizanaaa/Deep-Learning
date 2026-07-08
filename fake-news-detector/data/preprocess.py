"""
Preprocess the Fake and True news datasets.

Expected files:

data/raw/Fake.csv
data/raw/True.csv
"""

import pandas as pd

from sklearn.model_selection import train_test_split


def main():

    fake = pd.read_csv("data/raw/Fake.csv")

    true = pd.read_csv("data/raw/True.csv")

    fake["label"] = 1

    true["label"] = 0

    df = pd.concat([fake, true], ignore_index=True)

    df = df.sample(frac=1, random_state=42)

    df = df[["text", "label"]]

    train, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    train.to_csv(
        "data/processed/train.csv",
        index=False,
    )

    test.to_csv(
        "data/processed/test.csv",
        index=False,
    )

    print("Done")


if __name__ == "__main__":

    main()