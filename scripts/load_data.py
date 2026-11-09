"""
Load and prepare the financial sentiment dataset.

Loads the dataset from HuggingFace, splits into train/test, and saves the splits to data/train.csv
and data/test.csv.
"""

import os

import pandas as pd
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["prepare"]


def load_and_split():
    params = load_params()
    test_size = params["test_size"]
    random_seed = params["random_seed"]
    data_dir = params["data_dir"]
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    dataset_name = os.getenv("HF_DATASET_ID")
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name)

    # The dataset only has a train split
    # We create our own test split
    df = pd.DataFrame(dataset["train"]).rename(
        columns={"Title": "text", "Global Sentiment": "label"}
    )

    # Map integer labels to string labels matching FinBERT output
    label_map = {1: "positive", 0: "neutral", -1: "negative"}
    df["label"] = df["label"].map(label_map)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["label"],
    )

    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} samples saved in {train_path}")
    print(f"Test:  {len(test_df)} samples saved in {test_path}")
    print(
        f"Label distribution (train):\n{train_df['label'].value_counts().to_string()}"
    )


if __name__ == "__main__":
    load_and_split()
