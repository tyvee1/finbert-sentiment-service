"""
Model loading utilities.
"""

import os

from transformers import pipeline


def load_classifier():
    """
    Load the sentiment classifier.

    Controlled by the MODEL_SOURCE environment variable:
      - "mlflow" (default) — loads from the MLflow Model Registry
      - "huggingface"       — loads directly from HuggingFace Hub (used in CI/prod)
    """
    model_source = os.getenv("MODEL_SOURCE", "mlflow")

    if model_source == "huggingface":
        hf_model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
        print(f"Loading model from HuggingFace: {hf_model_id}")
        return pipeline(
            "text-classification",
            model=hf_model_id,
            tokenizer=hf_model_id,
            device="cpu",
        )

    import mlflow.transformers

    model_name = os.getenv("MODEL_NAME", "finbert")
    model_alias = os.getenv("MODEL_STAGE", "production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"Loading model from MLflow: models:/{model_name}@{model_alias}")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.transformers.load_model(f"models:/{model_name}@{model_alias}")
