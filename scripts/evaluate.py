"""
Evaluate the sentiment model on the held-out test set and register it in the
MLflow Model Registry.

Run from the project root:
    python scripts/evaluate.py

Exit codes (three values, not two — "could not check" must never read as "clean"):
    0  evaluation completed and the model was registered
    1  evaluation ran but a required step failed (e.g. registration rejected)
    2  evaluation could not run at all (missing test data, no tracking server)

--------------------------------------------------------------------------
Why `transformers` must stay below version 5 for this script to work
--------------------------------------------------------------------------
The starter pins ``transformers>=4.40.2`` with no upper bound. On a fresh
install that resolves to the 5.x line, and ``mlflow.transformers.log_model``
then fails: MLflow 2.16.2's transformers flavour imports
``FlaxPreTrainedModel``, which 5.x removed. The import failure makes MLflow's
framework detection fall through to TensorFlow, and the real error surfaces as
``ModuleNotFoundError: No module named 'tensorflow'`` — an error message that
points nowhere near the cause.

Reproduced both ways before the bound was added: identical code fails under
transformers 5.22.2 and succeeds under 4.57.6. requirements.txt now carries
``transformers>=4.40.2,<5``, and ``sentence-transformers<6`` with it, because
sentence-transformers 6 (pulled in by ``deepchecks[nlp]``) requires
transformers 5 and would otherwise drag the resolution back over the line.
"""

import os
import sys

import mlflow
import mlflow.transformers
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

load_dotenv()

# Exit codes, named so the call sites read as intent rather than as magic numbers.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_COULD_NOT_RUN = 2


def load_test_data(test_path: str) -> tuple[list[str], list[str]]:
    """Read the test split and return its texts and gold labels.

    Args:
        test_path: Path to the CSV produced by the DVC ``prepare`` stage.

    Returns:
        ``(texts, labels)`` as parallel lists.
    """
    df = pd.read_csv(test_path)
    return df["text"].tolist(), df["label"].tolist()


def build_classifier(model_id: str):
    """Build the HuggingFace text-classification pipeline under evaluation.

    ``truncation=True`` with ``max_length=512`` is set explicitly. Without it
    an input longer than the model's context raises instead of being truncated,
    so a single long row in the test set would end the evaluation rather than
    cost one prediction.

    Args:
        model_id: HuggingFace model repository id.

    Returns:
        A callable transformers pipeline.
    """
    print(f"Loading {model_id}...")
    return pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        truncation=True,
        max_length=512,
    )


def run_inference(classifier, texts: list[str]) -> list[str]:
    """Predict a label for every text.

    Args:
        classifier: The transformers pipeline.
        texts: Input strings.

    Returns:
        Predicted label strings, aligned with ``texts``.
    """
    print(f"Running inference on {len(texts)} samples...")
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute the four evaluation metrics.

    All three of precision, recall and F1 are weighted averages, which is the
    right choice for this dataset because its three classes are not balanced —
    a macro average would let the rarest class dominate the headline number,
    and a micro average would collapse to accuracy.

    ``zero_division=0`` is set so that a class the model never predicts
    contributes 0 rather than emitting a warning and a NaN. A NaN here would
    propagate into the promotion gate, and ``NaN >= threshold`` is False, so an
    unset metric would read as "below threshold" rather than as "not measured".

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.

    Returns:
        Dict with keys ``accuracy``, ``f1_weighted``, ``precision_weighted``
        and ``recall_weighted``.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def main() -> int:
    """Evaluate the model, log the run to MLflow, and register the model.

    Returns:
        A process exit code (see the module docstring).
    """
    model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
    model_name = os.getenv("MODEL_NAME", "finbert")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    test_path = os.path.join("data", "test.csv")

    if not os.path.exists(test_path):
        print(
            f"[COULD NOT RUN] {test_path} is missing.\n"
            "  CAUSE: the DVC `prepare` stage has not produced the test split.\n"
            "  REMEDY: run `dvc repro prepare` from the project root first.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_RUN

    texts, y_true = load_test_data(test_path)
    classifier = build_classifier(model_id)
    y_pred = run_inference(classifier, texts)
    metrics = compute_metrics(y_true, y_pred)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # ---------------------------------------------------------------------
    # MLflow tracking and registration
    # ---------------------------------------------------------------------
    # set_tracking_uri before set_experiment: set_experiment resolves (or
    # creates) the experiment against whatever store is currently configured,
    # so calling it first would create the experiment in a local ./mlruns
    # directory and then silently log the run somewhere else.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    try:
        with mlflow.start_run(run_name=f"eval-{model_id.split('/')[-1]}") as run:
            # Parameters: what was evaluated. Metrics: how it did. Keeping the
            # model id and the test-set size as params is what makes two runs
            # with different F1 comparable — without them a lower score is
            # indistinguishable from a different test split.
            mlflow.log_params(
                {
                    "model_id": model_id,
                    "task": "text-classification",
                    "max_length": 512,
                    "test_samples": len(texts),
                }
            )
            mlflow.log_metrics(metrics)

            # registered_model_name is what creates the registry entry and a
            # new version of it in the same call. await_registration_for is
            # left at its default so the call does not return until the version
            # is actually in the registry — promote.py runs immediately after
            # and would otherwise race a version that does not exist yet.
            model_info = mlflow.transformers.log_model(
                transformers_model=classifier,
                artifact_path="model",
                task="text-classification",
                registered_model_name=model_name,
                input_example=[texts[0]] if texts else ["Example headline."],
            )

            print(f"\nRun id:        {run.info.run_id}")
            print(f"Model URI:     {model_info.model_uri}")
            print(f"Registered as: {model_name}")
    except Exception as exc:  # noqa: BLE001 - reports the cause, then fails
        # Not caught to continue — caught to say what failed and WHY, then fail.
        # A bare `except: pass` here is how a registry ends up with no new
        # version while the script reports success.
        print(
            f"[FAILED] MLflow logging or registration failed: "
            f"{type(exc).__name__}: {exc}\n"
            f"  CAUSE: most often the tracking server at {tracking_uri} is not "
            "running, or `transformers` resolved to 5.x (see this module's "
            "docstring).\n"
            "  REMEDY: start it with `mlflow server --port 5000`, then confirm "
            "`pip show transformers` reports a 4.x version.",
            file=sys.stderr,
        )
        return EXIT_FAILED

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
