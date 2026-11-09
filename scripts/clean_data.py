"""
Clean raw Bluesky financial posts for use with FinBERT inference.

Input:  data/raw_stream.csv
Output: data/stream.csv
"""

import os
import re

import pandas as pd
import yaml


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["clean"]


# Only keep posts that contain financial keywords to filter out noise
FINANCIAL_KEYWORDS = {
    "stock",
    "stocks",
    "market",
    "markets",
    "share",
    "shares",
    "equity",
    "earnings",
    "revenue",
    "profit",
    "loss",
    "dividend",
    "ipo",
    "nasdaq",
    "nyse",
    "s&p",
    "dow",
    "trading",
    "investor",
    "investing",
    "portfolio",
    "fund",
    "etf",
    "bond",
    "bonds",
    "crypto",
    "bitcoin",
    "interest rate",
    "inflation",
    "gdp",
    "fed",
    "central bank",
    "quarter",
    "fiscal",
    "valuation",
    "bull",
    "bear",
    "rally",
    "selloff",
    "buyback",
}


def strip_emojis(text: str) -> str:
    return re.sub(r"[^\x00-\x7F]+", " ", text)


def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def strip_mentions(text: str) -> str:
    return re.sub(r"@\w+", "", text)


def strip_hashtag_symbols(text: str) -> str:
    # Remove # but keep the word.
    return re.sub(r"#(\w+)", r"\1", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def has_financial_keyword(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in FINANCIAL_KEYWORDS)


def clean(text: str) -> str:
    text = strip_urls(text)
    text = strip_mentions(text)
    text = strip_hashtag_symbols(text)
    text = strip_emojis(text)
    text = normalize_whitespace(text)
    return text


def clean_stream():
    params = load_params()
    min_words = params["min_words"]
    max_words = params["max_words"]
    data_dir = params["data_dir"]
    raw_path = os.path.join(data_dir, "raw_stream.csv")
    output_path = os.path.join(data_dir, "stream.csv")

    df = pd.read_csv(raw_path)
    print(f"Raw posts: {len(df)}")

    # clean the posts
    df["text"] = df["text"].astype(str).apply(clean)

    # drop duplicates
    df = df.drop_duplicates(subset="text")

    # drop too short posts
    df = df[df["text"].apply(lambda t: len(t.split()) >= min_words)]

    # truncate too long posts (FinBERT's max token limit is 512)
    df["text"] = df["text"].apply(lambda t: " ".join(t.split()[:max_words]))

    # keep only finance-related posts
    df = df[df["text"].apply(has_financial_keyword)]

    df = df.reset_index(drop=True)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Clean posts: {len(df)}")
    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    clean_stream()
