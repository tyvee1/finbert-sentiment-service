"""
Model loading utilities.
"""

import os

from transformers import pipeline

# The canonical label mapping for this checkpoint, in ONE place.
#
# The published model stores id2label as FLOATS — {"0": 0.0, "1": 1.0,
# "2": 2.0} — so a pipeline built from it returns `2.0` where a label belongs.
# scripts/smoke_test.py repairs the cached config.json on disk, and that is
# enough for a developer machine; it is NOT enough anywhere the cache is fresh.
# A container downloads its own copy at startup and nothing patches it, so the
# service answered 500 on every prediction while /health answered 200 — the
# model had loaded perfectly well.
#
# So the mapping lives here and the repair happens in-process at load, which
# holds regardless of what any cache contains. smoke_test.py imports these
# names rather than repeating them, because a mapping written down twice is a
# mapping that will eventually disagree with itself.
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def repair_label_mapping(classifier):
    """Give the model string labels if it came with numeric ones.

    Pipelines expose the mapping at ``classifier.model.config.id2label``. When
    every value is already a string the config is correct and is left exactly
    as it is — a loader that overwrote a correct mapping would silently impose
    this checkpoint's label order on any other model someone pointed the
    service at.

    Args:
        classifier: a transformers pipeline, or anything exposing
            ``.model.config``. Objects without that attribute are returned
            untouched, so a stub in a test is not required to imitate the whole
            pipeline shape.

    Returns:
        The same classifier, for use as ``return repair_label_mapping(...)``.
    """
    config = getattr(getattr(classifier, "model", None), "config", None)
    if config is None:
        return classifier

    id2label = getattr(config, "id2label", None)
    if not id2label:
        return classifier

    # Only repair a mapping that is actually broken. `isinstance(v, str)` is
    # the test, not "does it look like a number": a model whose labels are
    # genuinely "0" and "1" as strings is self-consistent and none of this
    # function's business.
    if all(isinstance(value, str) for value in id2label.values()):
        return classifier

    repaired = {}
    for key in id2label:
        # Keys arrive as ints from a live config and as strings from JSON.
        index = int(key)
        if index not in LABEL_NAMES:
            # An index this checkpoint does not define. Leave the whole mapping
            # alone rather than guess: a wrong label is worse than a clear
            # error, and run_predictions raises a message naming the cause.
            return classifier
        repaired[index] = LABEL_NAMES[index]

    config.id2label = repaired
    config.label2id = {name: index for index, name in repaired.items()}
    print(f"Repaired non-string id2label on the loaded model: {repaired}")
    return classifier


def load_classifier():
    """
    Load the sentiment classifier.

    Controlled by the MODEL_SOURCE environment variable:
      - "mlflow" (default) — loads from the MLflow Model Registry
      - "huggingface"       — loads directly from HuggingFace Hub (used in CI/prod)

    Either way the loaded model's label mapping is repaired if it needs it, so
    the service does not depend on a cache having been patched out of band.
    """
    model_source = os.getenv("MODEL_SOURCE", "mlflow")

    if model_source == "huggingface":
        hf_model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
        print(f"Loading model from HuggingFace: {hf_model_id}")
        return repair_label_mapping(
            pipeline(
                "text-classification",
                model=hf_model_id,
                tokenizer=hf_model_id,
                device="cpu",
            )
        )

    import mlflow.transformers

    model_name = os.getenv("MODEL_NAME", "finbert")
    model_alias = os.getenv("MODEL_STAGE", "production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"Loading model from MLflow: models:/{model_name}@{model_alias}")
    mlflow.set_tracking_uri(tracking_uri)
    return repair_label_mapping(
        mlflow.transformers.load_model(f"models:/{model_name}@{model_alias}")
    )
