"""
Assign the ``candidate`` alias to a registered model version, for A/B testing.

WHY THIS EXISTS SEPARATELY FROM promote.py
    ``promote.py`` decides whether a newly evaluated model is good enough to
    serve, and moves the ``production`` alias only when it clears the
    ``f1_weighted`` threshold in params.yaml. That gate is the point of it.

    A candidate is a different question. You are not asking "is this good
    enough to serve?" — you are asking "is this better than what is serving?",
    and the way to find out is to route some traffic to it and compare. Running
    a candidate through the production gate would reject exactly the models an
    A/B test exists to evaluate: the ones you are not yet sure about.

    So this assigns an alias without a quality gate, and says so plainly rather
    than pretending the absence of a check is an oversight.

USAGE
    # the newest registered version becomes the candidate
    python scripts/set_candidate_alias.py

    # or a specific one
    python scripts/set_candidate_alias.py --version 9

    # clear it, so no candidate is routable
    python scripts/set_candidate_alias.py --clear

    Then restart the API. The candidate is loaded at startup, not per request:
    loading a transformer takes seconds and doing it inside a request handler
    would make the first request after a change time out.

WHAT IT REFUSES TO DO
    It will not point the candidate alias at the version ``production``
    currently resolves to. An A/B test whose two arms are the same model
    produces a difference of zero, reports it with full confidence, and looks
    exactly like a successful experiment showing no regression.
"""

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign or clear the A/B candidate alias."
    )
    parser.add_argument(
        "--version",
        help="Registered version to make the candidate. Defaults to the "
             "highest version number in the registry.",
    )
    parser.add_argument(
        "--alias",
        default=os.getenv("AB_CANDIDATE_ALIAS", "candidate"),
        help="Alias to assign (default: candidate, or AB_CANDIDATE_ALIAS).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "finbert"),
        help="Registered model name (default: finbert, or MODEL_NAME).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove the alias instead of assigning it.",
    )
    parser.add_argument(
        "--allow-same-as-production",
        action="store_true",
        help="Permit pointing the candidate at the serving version. Only "
             "useful for testing the routing itself, never for a comparison.",
    )
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if args.clear:
        try:
            client.delete_registered_model_alias(args.model, args.alias)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"could not clear alias {args.alias!r}: "
                  f"{type(exc).__name__}: {exc}")
            return 1
        print(f"cleared alias {args.alias!r} on model {args.model!r}; "
              f"no candidate is routable until one is set again")
        return 0

    version = args.version
    if not version:
        versions = client.search_model_versions(f"name='{args.model}'")
        if not versions:
            print(f"no registered versions for model {args.model!r} — run "
                  f"`python scripts/evaluate.py` first")
            return 1
        # Sorted numerically: string ordering puts version 10 before version 9.
        version = str(max(int(v.version) for v in versions))

    production_alias = os.getenv("MODEL_STAGE", "production")
    try:
        serving = client.get_model_version_by_alias(args.model, production_alias)
        serving_version = serving.version
    except Exception:  # noqa: BLE001 - no production alias yet is fine
        serving_version = None

    if serving_version == version and not args.allow_same_as_production:
        print(
            f"refusing: version {version} is what {production_alias!r} already "
            f"resolves to, so both arms of the A/B test would be the same "
            f"model. That produces a difference of zero and reports it as a "
            f"result. Pass --version with a different one, or "
            f"--allow-same-as-production if you are testing the routing "
            f"itself rather than comparing models."
        )
        return 1

    client.set_registered_model_alias(args.model, args.alias, version)
    print(f"alias {args.alias!r} -> {args.model} version {version}")
    if serving_version:
        print(f"  {production_alias!r} is version {serving_version}, so the "
              f"comparison is {serving_version} (production) vs {version} "
              f"(candidate)")
    print("  restart the API for it to be loaded, then check GET /ab-status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
