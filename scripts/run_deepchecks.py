"""
Drift gate: compares the live stream against the evaluation reference set and
fails the build when either drift measure exceeds its threshold.

Run from the project root:
    python scripts/run_deepchecks.py

Exit codes (three values, and the third is the whole point):
    0  both drift scores are within their thresholds
    1  at least one drift score exceeded its threshold — the gate FIRED
    2  the gate could not run, so nothing was measured

Code 2 exists because the alternative is a gate that reports success when it
failed to look. Every "could not" path below — missing data, too few samples,
an unreachable model download — returns 2, and the CI job treats 2 as a failure
of the build rather than as a pass.

--------------------------------------------------------------------------
Three things about this gate that are not obvious from the specification
--------------------------------------------------------------------------

1.  IT NEEDS THE INTERNET, AND NOT ONLY FOR THE MODEL.
    ``TextData.calculate_builtin_properties`` calls ``get_fasttext_model()``
    unconditionally — the call sits outside any ``if``, so it happens even when
    only the ``Sentiment`` property is requested. That downloads
    ``lid.176.bin`` (~126 MB) from ``dl.fbaipublicfiles.com``, a host unrelated
    to HuggingFace, PyPI or AWS. Verified by reading deepchecks 0.19.1's
    ``text_properties.py`` and by the download failing in a network-restricted
    environment. ``models_storage`` is therefore pointed at a cache directory
    the CI job caches, so the download happens once rather than per run, and a
    download failure is reported as code 2.

2.  DRIFT NEEDS AT LEAST 100 SAMPLES PER SIDE.
    deepchecks raises ``NotEnoughSamplesError`` below 100, which an
    ``except Exception`` around the check would turn into either a false "drift
    detected" or a false "clean". It is caught by name and reported as code 2
    with both row counts printed.

3.  WHAT THIS GATE ACTUALLY MEASURES.
    The reference side is ``data/test.csv`` — curated, labelled financial
    headlines. The current side is ``data/stream.csv`` — cleaned Bluesky posts.
    These are different populations collected different ways, so a high score
    here may be a faithful measurement of a difference that was there on day
    one rather than of a model degrading over time. The number is still worth
    gating on, because a change in it is informative; the absolute value is
    not evidence of decay. Stated here so the gate is not read as more than it
    is: this same confound is what made an earlier lesson in this course
    compare two cities in the same month and call the difference drift.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from deepchecks.core.errors import NotEnoughSamplesError
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

from app.utils import load_classifier

load_dotenv()

EXIT_NO_DRIFT = 0
EXIT_DRIFT_DETECTED = 1
EXIT_COULD_NOT_CHECK = 2

# The property the specification names. Requesting only this one keeps the
# computation to a text-sentiment pass instead of the full default property
# suite, which would additionally download toxicity, fluency and formality
# transformer models.
DRIFT_PROPERTY = "Sentiment"

# Where deepchecks caches lid.176.bin. Kept inside the repo working tree (and
# gitignored) so the CI cache action has a stable path to restore into.
MODELS_STORAGE = os.path.join(".deepchecks_models")

# deepchecks' own documented floor for a drift calculation.
MIN_DRIFT_SAMPLES = 100


def load_params() -> dict:
    """Read the ``deepchecks`` section of params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    """Predict a label for every text.

    Args:
        classifier: The loaded text-classification pipeline.
        texts: Input strings.

    Returns:
        Predicted label strings aligned with ``texts``.
    """
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def build_text_data(texts: list[str]) -> TextData:
    """Wrap texts in a ``TextData`` and compute the drift property on it.

    Args:
        texts: Input strings.

    Returns:
        A ``TextData`` with the ``Sentiment`` property calculated.

    Raises:
        RuntimeError: if the property calculation could not complete — most
            often because the fastText language model could not be fetched.
            Re-raised as one named type so the caller can report code 2 without
            an ``except Exception`` that would also swallow real bugs.
    """
    dataset = TextData(raw_text=texts, task_type="text_classification")
    try:
        dataset.calculate_builtin_properties(
            include_properties=[DRIFT_PROPERTY],
            models_storage=MODELS_STORAGE,
            cache_models=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not calculate the {DRIFT_PROPERTY!r} property: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return dataset


def main() -> int:
    """Run both drift checks and return the gate's verdict as an exit code."""
    try:
        params = load_params()
        property_drift_threshold = float(params["property_drift_threshold"])
        prediction_drift_threshold = float(params["prediction_drift_threshold"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(
            f"[COULD NOT CHECK] cannot read the deepchecks thresholds from "
            f"params.yaml: {type(exc).__name__}: {exc}\n"
            "  REMEDY: run this script from the project root.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_CHECK

    stream_path = os.path.join("data", "stream.csv")
    test_path = os.path.join("data", "test.csv")
    missing = [p for p in (stream_path, test_path) if not os.path.exists(p)]
    if missing:
        print(
            f"[COULD NOT CHECK] missing input data: {', '.join(missing)}\n"
            "  CAUSE: the DVC pipeline has not been run in this working tree.\n"
            "  REMEDY: `dvc repro` from the project root.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_CHECK

    stream_df = pd.read_csv(stream_path)
    test_df = pd.read_csv(test_path)
    stream_texts = stream_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    # Checked before the expensive work rather than after: deepchecks would
    # raise here anyway, but only after a full property pass and a full
    # inference pass had already been paid for.
    too_small = {
        name: len(rows)
        for name, rows in (("data/test.csv", test_texts), ("data/stream.csv", stream_texts))
        if len(rows) < MIN_DRIFT_SAMPLES
    }
    if too_small:
        detail = ", ".join(f"{k} has {v} rows" for k, v in too_small.items())
        print(
            f"[COULD NOT CHECK] drift needs at least {MIN_DRIFT_SAMPLES} "
            f"samples per dataset; {detail}.\n"
            "  Nothing has been measured, so this is NOT a pass.\n"
            "  REMEDY: widen the clean.min_words / clean.max_words window in "
            "params.yaml, or supply more raw stream data.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_CHECK

    print("Loading production model...")
    try:
        classifier = load_classifier()
    except Exception as exc:  # noqa: BLE001 - reported as code 2, never as clean
        print(
            f"[COULD NOT CHECK] the model would not load: "
            f"{type(exc).__name__}: {exc}\n"
            f"  MODEL_SOURCE is {os.getenv('MODEL_SOURCE', 'mlflow')!r}.\n"
            "  REMEDY: set MODEL_SOURCE=huggingface to bypass the registry, or "
            "start the MLflow server and promote a version first.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_CHECK

    # reference = the curated evaluation set; current = the live-like stream.
    # deepchecks names these train_dataset and test_dataset, which is the
    # opposite of what "test" means in this project's file names, so the local
    # variables say reference/current and the mapping is made once, here.
    print(f"Calculating the {DRIFT_PROPERTY!r} property on both datasets...")
    try:
        reference_ds = build_text_data(test_texts)
        current_ds = build_text_data(stream_texts)
    except RuntimeError as exc:
        print(
            f"[COULD NOT CHECK] {exc}\n"
            "  CAUSE: deepchecks downloads lid.176.bin from "
            "dl.fbaipublicfiles.com on first use, unconditionally, even for a "
            "single property.\n"
            "  Nothing has been measured, so this is NOT a pass.\n"
            f"  REMEDY: allow egress to that host, or restore a warm "
            f"{MODELS_STORAGE}/ cache.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_CHECK

    failures: list[str] = []

    # ---------------------------------------------------------------- property
    try:
        property_result = PropertyDrift().run(
            train_dataset=reference_ds, test_dataset=current_ds
        )
    except NotEnoughSamplesError as exc:
        print(f"[COULD NOT CHECK] property drift: {exc}", file=sys.stderr)
        return EXIT_COULD_NOT_CHECK

    # Shape confirmed against deepchecks 0.19.1 by execution:
    #   {"Sentiment": {"Drift score": float, "Method": str, "Importance": None}}
    if DRIFT_PROPERTY not in property_result.value:
        print(
            f"[COULD NOT CHECK] property drift returned no {DRIFT_PROPERTY!r} "
            f"entry; it reported {sorted(property_result.value)}.\n"
            "  Nothing has been measured for the gated property.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_CHECK

    property_score = float(property_result.value[DRIFT_PROPERTY]["Drift score"])
    property_method = property_result.value[DRIFT_PROPERTY]["Method"]
    print(
        f"\n{DRIFT_PROPERTY} property drift : {property_score:.4f} "
        f"(threshold {property_drift_threshold:.4f}, method {property_method})"
    )
    if property_score > property_drift_threshold:
        failures.append(
            f"{DRIFT_PROPERTY} property drift {property_score:.4f} exceeds "
            f"{property_drift_threshold:.4f}"
        )

    # -------------------------------------------------------------- prediction
    # Both checks are evaluated before exiting, rather than returning at the
    # first failure as the starter's comments suggest. One run then reports both
    # numbers: knowing whether the inputs drifted, the outputs drifted, or both
    # is the difference between a data problem and a model problem, and a gate
    # that stops at the first failure makes the operator run it twice to find
    # out. The contract is unchanged — non-zero when either exceeds.
    print("Running predictions on both datasets for prediction drift...")
    try:
        reference_predictions = run_predictions(classifier, test_texts)
        current_predictions = run_predictions(classifier, stream_texts)
    except Exception as exc:  # noqa: BLE001 - reported as code 2, never as clean
        print(
            f"[COULD NOT CHECK] inference failed while computing prediction "
            f"drift: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_CHECK

    try:
        prediction_result = PredictionDrift().run(
            train_dataset=reference_ds,
            test_dataset=current_ds,
            train_predictions=reference_predictions,
            test_predictions=current_predictions,
        )
    except NotEnoughSamplesError as exc:
        print(f"[COULD NOT CHECK] prediction drift: {exc}", file=sys.stderr)
        return EXIT_COULD_NOT_CHECK

    # Shape confirmed by execution:
    #   {"Drift score": float, "Method": str, "Samples per class": {...}}
    prediction_score = float(prediction_result.value["Drift score"])
    prediction_method = prediction_result.value["Method"]
    print(
        f"Prediction drift         : {prediction_score:.4f} "
        f"(threshold {prediction_drift_threshold:.4f}, method {prediction_method})"
    )
    if prediction_score > prediction_drift_threshold:
        failures.append(
            f"prediction drift {prediction_score:.4f} exceeds "
            f"{prediction_drift_threshold:.4f}"
        )

    # ------------------------------------------------------------------ verdict
    if failures:
        print("\n[DRIFT DETECTED] the gate has fired:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\n  This blocks the build. Before raising a threshold, check "
            "whether the reference and current datasets are the same kind of "
            "text at all — see note 3 in this script's docstring.",
            file=sys.stderr,
        )
        return EXIT_DRIFT_DETECTED

    print("\n[OK] both drift scores are within their thresholds.")
    return EXIT_NO_DRIFT


if __name__ == "__main__":
    sys.exit(main())
