"""
Promote the latest registered model version to the ``production`` alias, but
only if the most recent evaluation run cleared the F1 threshold.

Run from the project root, after scripts/evaluate.py:
    python scripts/promote.py

Exit codes (three values — a gate that cannot decide must not report "clean"):
    0  threshold met and the alias was assigned, OR threshold not met and the
       model was correctly left unpromoted. Both are successful gate decisions.
    1  the gate reached a decision it refuses to act on — the latest registered
       version is not the one that was measured (see the TOCTOU note below).
    2  the gate could not decide: no experiment, no runs, or no f1_weighted
       metric on the run it found. Nothing is promoted.

--------------------------------------------------------------------------
The two failure modes this script is written to avoid
--------------------------------------------------------------------------

1.  FAILING OPEN. The tempting shape for a promotion gate is

        try:
            ... look up the score ...
        except Exception:
            return True        # "be permissive, it is only coursework"

    which promotes an unmeasured model every time the tracking server is
    unreachable. The same pattern appears in this course's own lesson 10
    material. Here every lookup failure returns exit code 2 and promotes
    nothing. "I could not measure it" and "it measured well" are different
    answers and only one of them is allowed to promote.

2.  PROMOTING A VERSION THAT WAS NEVER MEASURED. The specification describes
    two separate lookups — the most recent evaluation RUN for the score, and
    the latest registered VERSION for the alias — and assumes they refer to the
    same model. They usually do, and nothing guarantees it: a second
    ``evaluate.py`` finishing between the two reads, or a version registered by
    hand, breaks the link. The alias would then point at a version whose score
    was never checked, which is precisely the failure the gate exists to
    prevent.

    MLflow records the originating ``run_id`` on every model version, so the
    link is checkable. This script reads the latest version as specified, then
    asserts its ``run_id`` matches the run it scored. On a mismatch it promotes
    nothing and exits 1 with both ids printed, rather than guessing which of
    the two the operator meant.
"""

import math
import os
import sys

import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_COULD_NOT_DECIDE = 2

PRODUCTION_ALIAS = "production"


def load_params() -> dict:
    """Read the ``promote`` section of params.yaml.

    The threshold lives in params.yaml rather than in this file so that DVC
    sees it as a tracked parameter and so that changing the production bar is a
    reviewable one-line diff rather than a code change.
    """
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    """Return the run id and ``f1_weighted`` of the most recent evaluation run.

    Args:
        client: An MLflow client pointed at the tracking server.
        experiment_name: Experiment the evaluation run was logged to.

    Returns:
        ``(run_id, f1_weighted)``.

    Raises:
        LookupError: if the experiment does not exist, contains no runs, or the
            most recent run carries no ``f1_weighted`` metric. Raised rather
            than returning a sentinel such as ``0.0`` or ``None``: a sentinel
            would flow into the comparison below and be reported as a score
            that failed the threshold, which is a different and much more
            misleading statement than "there is no score".
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise LookupError(
            f"experiment {experiment_name!r} does not exist on this tracking server"
        )

    # order_by start_time DESC rather than relying on the default ordering:
    # MLflow's default sort is not documented as newest-first, and a gate that
    # silently scores an old run is worse than one that fails.
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"experiment {experiment_name!r} contains no runs")

    run = runs[0]
    if "f1_weighted" not in run.data.metrics:
        raise LookupError(
            f"run {run.info.run_id} has no 'f1_weighted' metric "
            f"(metrics present: {sorted(run.data.metrics) or 'none'})"
        )

    f1 = float(run.data.metrics["f1_weighted"])
    if math.isnan(f1):
        # NaN >= threshold evaluates to False, so a NaN would otherwise be
        # reported as "below threshold" — an unmeasured model described as a
        # measured and rejected one.
        raise LookupError(f"run {run.info.run_id} recorded f1_weighted as NaN")

    return run.info.run_id, f1


def get_latest_version(client: MlflowClient, model_name: str):
    """Return the highest-numbered registered version of ``model_name``.

    Version numbers are compared as integers. MLflow returns them as strings,
    and a lexicographic max over ``["9", "10"]`` is ``"9"`` — so a registry
    that has passed its ninth version would silently stop advancing.

    Args:
        client: An MLflow client.
        model_name: Registered model name.

    Returns:
        The latest ``ModelVersion``.

    Raises:
        LookupError: if the model is not registered or has no versions.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise LookupError(f"no registered versions found for model {model_name!r}")
    return max(versions, key=lambda v: int(v.version))


def promote(client: MlflowClient, model_name: str, expected_run_id: str):
    """Assign the ``production`` alias to the latest version of the model.

    Args:
        client: An MLflow client.
        model_name: Registered model name.
        expected_run_id: The run whose score authorised this promotion. The
            latest version's own ``run_id`` must match it.

    Returns:
        The promoted ``ModelVersion``.

    Raises:
        LookupError: if the model has no versions.
        RuntimeError: if the latest version did not come from
            ``expected_run_id`` — see failure mode 2 in the module docstring.
    """
    version = get_latest_version(client, model_name)

    if version.run_id != expected_run_id:
        raise RuntimeError(
            "refusing to promote: the latest registered version was not "
            "produced by the run that was scored.\n"
            f"  scored run        : {expected_run_id}\n"
            f"  version {version.version} came from: {version.run_id}\n"
            "  CAUSE: another evaluation registered a version between the "
            "score lookup and this promotion, or a version was registered "
            "outside evaluate.py.\n"
            "  REMEDY: re-run `python scripts/evaluate.py` then this script, "
            "so the scored run and the latest version are the same model."
        )

    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version.version)
    return version


def main() -> int:
    """Decide and apply (or withhold) promotion.

    Returns:
        A process exit code (see the module docstring).
    """
    model_name = os.getenv("MODEL_NAME", "finbert")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    try:
        threshold = float(load_params()["f1_threshold"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(
            f"[COULD NOT DECIDE] cannot read promote.f1_threshold from "
            f"params.yaml: {type(exc).__name__}: {exc}\n"
            "  REMEDY: run this script from the project root, where "
            "params.yaml lives.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_DECIDE

    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        run_id, f1 = get_latest_f1(client, experiment_name)
    except LookupError as exc:
        print(
            f"[COULD NOT DECIDE] no usable evaluation score: {exc}\n"
            "  Nothing has been promoted.\n"
            "  REMEDY: run `python scripts/evaluate.py` first.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_DECIDE
    except Exception as exc:  # noqa: BLE001 - any failure here must NOT promote
        print(
            f"[COULD NOT DECIDE] could not reach the tracking server at "
            f"{tracking_uri}: {type(exc).__name__}: {exc}\n"
            "  Nothing has been promoted.\n"
            "  REMEDY: start it with `mlflow server --port 5000`.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_DECIDE

    print(f"Latest evaluation run: {run_id}")
    print(f"  f1_weighted : {f1:.4f}")
    print(f"  threshold   : {threshold:.4f}")

    # ">=" and not ">": the specification says "meets or exceeds", and a model
    # scoring exactly the threshold has met the bar it was set.
    if f1 < threshold:
        print(
            f"\nNOT PROMOTED: f1_weighted {f1:.4f} is below the required "
            f"threshold of {threshold:.4f}.\n"
            f"The '{PRODUCTION_ALIAS}' alias has been left unchanged, so "
            "whatever is currently serving keeps serving."
        )
        # A correct decision, correctly applied: exit 0. Returning non-zero
        # here would make every CI run that legitimately rejects a weak model
        # indistinguishable from a broken gate.
        return EXIT_OK

    try:
        version = promote(client, model_name, expected_run_id=run_id)
    except RuntimeError as exc:
        print(f"\n[REFUSED] {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except LookupError as exc:
        print(
            f"\n[COULD NOT DECIDE] {exc}\n"
            "  REMEDY: run `python scripts/evaluate.py`, which registers the "
            "model as part of logging it.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_DECIDE

    print(
        f"\nPROMOTED: f1_weighted {f1:.4f} meets the threshold "
        f"{threshold:.4f}.\n"
        f"Model '{model_name}' version {version.version} now carries the "
        f"'{PRODUCTION_ALIAS}' alias."
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
