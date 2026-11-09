"""
Smoke test to confirm the model loads correctly
and runs a basic inference.

Also patches the model's config.json in the HuggingFace cache to fix a known
issue where id2label values are stored as floats instead of strings.

Run:
    python smoke_test.py
"""

import json
import os
import sys


def download_and_patch_model(model_id: str) -> None:
    """
    Download the model and fix the broken id2label in the config.
    The model was saved with float values (0.0, 1.0, 2.0) instead of string labels,
    which causes newer versions of transformers to reject the config.
    """
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    print(config_path)
    with open(config_path) as f:
        config = json.load(f)

    # The mapping comes from app/utils.py so there is one definition of it.
    # This script repairs the cache on disk; the service repairs the loaded
    # model in memory. Two repairs disagreeing about which index is "positive"
    # would be a silent mislabelling, so neither spells the mapping out.
    try:
        from app.utils import LABEL_NAMES
    except ImportError:  # pragma: no cover - running from inside scripts/
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.utils import LABEL_NAMES

    config["id2label"] = {str(i): name for i, name in LABEL_NAMES.items()}
    config["label2id"] = {name: str(i) for i, name in LABEL_NAMES.items()}

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Patched config.json for {model_id}")


def check_model():
    model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
    print(f"Loading {model_id} model...")
    try:
        download_and_patch_model(model_id)

        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model=model_id,
            tokenizer=model_id,
        )
        result = classifier("The company reported record profits this quarter.")
        assert result and result[0]["label"] in {"positive", "negative", "neutral"}
        print(f"Model loaded and inference OK  -->  {result[0]}")
        return True
    except Exception as e:  # noqa: BLE001 - a smoke test reports any failure
        print(f"[FAIL] Model check failed: {e}")
        return False


def main():
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 50)
    print("  FinBERT Market Smoke Test")
    print("=" * 50)

    passed = check_model()

    print("=" * 50)
    if passed:
        print("  All checks passed.")
        sys.exit(0)
    else:
        print("  Smoke test FAILED. Fix the errors above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
