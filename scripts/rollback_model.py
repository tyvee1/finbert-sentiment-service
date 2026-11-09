"""
Roll the ``production`` alias back to the previously promoted model version.

Run from the project root:
    python scripts/rollback_model.py --dry-run      # show what would change
    python scripts/rollback_model.py                # do it

Exit codes:
    0  the alias was moved (or, with --dry-run, would be)
    1  rollback was refused — there is no safe previous version to go back to
    2  it could not decide: no registry, no tracking server, no alias

--------------------------------------------------------------------------
What "the previous version" means, and why it is not version N-1
--------------------------------------------------------------------------
The obvious implementation reads the current alias target, subtracts one, and
sets the alias to that. It is wrong in two ways that only show up in the state
you need a rollback in:

  * Version numbers are not a promotion history. Versions get registered by
    evaluation runs that never passed the threshold, so N-1 can easily be a
    version that was deliberately rejected. Rolling onto it promotes a model
    that was already judged unfit.

  * Version numbers have gaps. A deleted version leaves a hole, and N-1 may
    not exist at all.

So this script rolls back to the most recent version that was itself ever
aliased ``production`` — recorded as a tag on the version at promotion time,
because MLflow's registry keeps no alias history of its own. If no such
version exists, there is nothing safe to roll back to and the script refuses
rather than picking the nearest number.

--------------------------------------------------------------------------
How it is triggered
--------------------------------------------------------------------------
Manually, or by the rollback workflow in .github/workflows/rollback.yml. See
the "Model rollback" section of the README for both paths and for the limits
of the automatic one.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_COULD_NOT_DECIDE = 2

PRODUCTION_ALIAS = "production"
# Set on a version when it is promoted, so that promotion history survives
# the alias being moved. An alias records only where it points now.
PROMOTED_TAG = "was_promoted"


def promoted_versions(client: MlflowClient, model_name: str) -> list:
    """Every version ever promoted, newest first.

    Ordered by integer version number, not lexicographically: ``max("9", "10")``
    is ``"9"``, so a registry past its ninth version would silently stop
    advancing.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    promoted = [v for v in versions if PROMOTED_TAG in (v.tags or {})]
    return sorted(promoted, key=lambda v: int(v.version), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report the change without making it")
    args = parser.parse_args()

    model_name = os.getenv("MODEL_NAME", "finbert")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        current = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except Exception as exc:  # noqa: BLE001 - any failure here must not move the alias
        print(
            f"[COULD NOT DECIDE] no '{PRODUCTION_ALIAS}' alias to roll back from: "
            f"{type(exc).__name__}: {exc}\n"
            f"  The alias has not been changed.\n"
            f"  REMEDY: confirm the tracking server at {tracking_uri} is "
            f"reachable and that '{model_name}' has a promoted version.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_DECIDE

    history = promoted_versions(client, model_name)
    candidates = [v for v in history if int(v.version) != int(current.version)]

    if not candidates:
        print(
            f"[REFUSED] version {current.version} is the only version that has "
            f"ever carried the '{PRODUCTION_ALIAS}' alias, so there is no "
            "previously-approved model to roll back to.\n"
            "  The alias has NOT been moved. Rolling onto an arbitrary earlier "
            "version number would promote a model that may never have passed "
            "the threshold.\n"
            "  REMEDY: fix forward — investigate the drift, re-evaluate, and "
            "promote a version that clears the gate.",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    target = candidates[0]
    print(f"Model      : {model_name}")
    print(f"Current    : version {current.version}")
    print(f"Rolling to : version {target.version} (previously promoted)")

    if args.dry_run:
        print("\n--dry-run: the alias has not been changed.")
        return EXIT_OK

    client.set_registered_model_alias(
        model_name, PRODUCTION_ALIAS, target.version
    )
    # Record why, on the version itself. An alias move leaves no trace, and
    # "why is production on an older version?" is the first question anyone
    # asks afterwards.
    client.set_model_version_tag(
        model_name, target.version, "rolled_back_from", str(current.version)
    )
    print(
        f"\nROLLED BACK: '{PRODUCTION_ALIAS}' now points at version "
        f"{target.version}.\n"
        f"Version {current.version} keeps its '{PROMOTED_TAG}' tag, so a "
        "later rollback can still find it."
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
