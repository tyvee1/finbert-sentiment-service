"""
Production monitoring stream for the FinBERT sentiment API.

Reads headlines from data/stream.csv, sends them to the /predict endpoint, and
logs aggregated metrics to MLflow every WINDOW_SIZE predictions.

Each observation window logs:
    - Sentiment distribution (% positive, % negative, % neutral)
    - Average confidence score
    - Average latency (ms)

Run from the project root, with the API already serving:
    python monitoring/stream.py

Exit codes:
    0  the stream completed and every window was logged
    1  the stream stopped early (the API became unreachable mid-run)
    2  it could not start at all (no stream data, no API, no tracking server)

--------------------------------------------------------------------------
Why windows, and why the step number matters
--------------------------------------------------------------------------
A single aggregate over the whole stream would answer "what did the model do?"
and nothing else. Windowing turns the same data into a time series, and the
``step`` argument is what makes it one: MLflow stores a metric as a history of
(step, value) points, so logging ``pct_positive`` at step 0, 1, 2... produces a
curve. Logging it repeatedly with no step overwrites the same point and
produces a single number that looks like a measurement and carries no trend —
which is the difference between monitoring and reporting.

That curve is the thing a production alert reads. A model whose ``pct_positive``
drifts from 0.3 to 0.8 over a shift has changed behaviour even if every
individual prediction looks confident and well-formed, and no per-request check
can see it.

A note on the last window: the final batch is almost never a full WINDOW_SIZE.
It is logged anyway, because dropping it silently loses real traffic, but its
percentages rest on fewer samples and are correspondingly noisier. The run
prints the sample count for every window so a reader can see which one that is.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# localhost rather than API_HOST: API_HOST is the address the server BINDS to,
# and its default of 0.0.0.0 means "every interface". As a destination to
# connect to, 0.0.0.0 is not a routable address on macOS, so reusing it here
# would make the client fail against a server that is running correctly.
API_HOST = os.getenv("MONITOR_API_HOST", "localhost")
API_URL = f"http://{API_HOST}:{os.getenv('API_PORT', 8000)}"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = 50  # number of predictions per observation window
SLEEP_MS = 100  # delay between requests to simulate real traffic (ms)

SENTIMENTS = ("positive", "negative", "neutral")

EXIT_OK = 0
EXIT_STOPPED_EARLY = 1
EXIT_COULD_NOT_START = 2


def predict(text: str) -> dict:
    """POST one headline to /predict and return the parsed result.

    Args:
        text: The headline to score.

    Returns:
        The decoded JSON body: ``text``, ``sentiment``, ``confidence``,
        ``latency_ms``.

    Raises:
        requests.HTTPError: on a non-2xx response.
        requests.RequestException: on a transport failure.
    """
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def log_window(window: list[dict], window_idx: int) -> None:
    """Aggregate one observation window and log it to MLflow.

    Logs five metrics at ``step=window_idx``:
    ``pct_positive``, ``pct_negative``, ``pct_neutral``, ``avg_confidence``,
    ``avg_latency_ms``.

    The three percentages are computed over all three labels explicitly rather
    than from whichever labels happen to appear in the window. A window with no
    negative predictions must log ``pct_negative = 0.0``, not omit it: an
    absent point in a metric history is indistinguishable from the monitor
    having been down, and "no negatives this window" is a signal, not a gap.

    Args:
        window: Prediction results collected for this window.
        window_idx: Zero-based window number, used as the MLflow step.

    Returns:
        None.
    """
    if not window:
        return

    size = len(window)
    counts = {s: 0 for s in SENTIMENTS}
    for result in window:
        label = result["sentiment"]
        # An unexpected label is counted separately rather than dropped — if
        # the model starts emitting something outside the known set, the
        # percentages must not quietly renormalise to hide it.
        counts[label] = counts.get(label, 0) + 1

    metrics = {
        "pct_positive": 100.0 * counts.get("positive", 0) / size,
        "pct_negative": 100.0 * counts.get("negative", 0) / size,
        "pct_neutral": 100.0 * counts.get("neutral", 0) / size,
        "avg_confidence": sum(r["confidence"] for r in window) / size,
        "avg_latency_ms": sum(r["latency_ms"] for r in window) / size,
    }

    mlflow.log_metrics(metrics, step=window_idx)

    unexpected = {k: v for k, v in counts.items() if k not in SENTIMENTS and v}
    note = f"  UNEXPECTED LABELS: {unexpected}" if unexpected else ""
    print(
        f"window {window_idx:>3}  n={size:<3}  "
        f"pos={metrics['pct_positive']:5.1f}%  "
        f"neg={metrics['pct_negative']:5.1f}%  "
        f"neu={metrics['pct_neutral']:5.1f}%  "
        f"conf={metrics['avg_confidence']:.3f}  "
        f"latency={metrics['avg_latency_ms']:.1f}ms{note}"
    )


def main() -> int:
    """Replay the cleaned stream against the API, logging one window at a time.

    Returns:
        A process exit code (see the module docstring).
    """
    stream_path = os.path.join("data", "stream.csv")
    if not os.path.exists(stream_path):
        print(
            f"[COULD NOT START] {stream_path} is missing.\n"
            "  CAUSE: the DVC `clean` stage has not run in this working tree.\n"
            "  REMEDY: `dvc repro clean` from the project root.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_START

    headlines = pd.read_csv(stream_path)["text"].astype(str).tolist()
    if not headlines:
        print(f"[COULD NOT START] {stream_path} has no rows.", file=sys.stderr)
        return EXIT_COULD_NOT_START

    # Fail fast and say which of the two services is missing, rather than
    # discovering it on the first prediction 30 seconds in.
    try:
        health = requests.get(f"{API_URL}/health", timeout=10)
    except requests.RequestException as exc:
        print(
            f"[COULD NOT START] the API at {API_URL} is not reachable: "
            f"{type(exc).__name__}: {exc}\n"
            "  REMEDY: start it with `python app/main.py`, or "
            "`docker compose up`.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_START
    if health.status_code != 200:
        print(
            f"[COULD NOT START] the API answered /health with "
            f"{health.status_code}, so no model is loaded.\n"
            "  REMEDY: check the API's logs for the model load error.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_START

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(
        f"Streaming {len(headlines)} headlines to {API_URL} "
        f"in windows of {WINDOW_SIZE}..."
    )

    exit_code = EXIT_OK
    with mlflow.start_run(run_name="production-monitoring"):
        mlflow.log_params(
            {
                "window_size": WINDOW_SIZE,
                "sleep_ms": SLEEP_MS,
                "stream_rows": len(headlines),
                "api_url": API_URL,
            }
        )

        window: list[dict] = []
        window_idx = 0
        for headline in headlines:
            try:
                window.append(predict(headline))
            except requests.RequestException as exc:
                # Flush what was collected before stopping: a partial window is
                # data, and discarding it would make the metric history end
                # earlier than the outage actually did.
                print(
                    f"\n[STOPPED EARLY] the API stopped responding after "
                    f"{window_idx * WINDOW_SIZE + len(window)} predictions: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                log_window(window, window_idx)
                exit_code = EXIT_STOPPED_EARLY
                break

            if len(window) >= WINDOW_SIZE:
                log_window(window, window_idx)
                window = []
                window_idx += 1

            time.sleep(SLEEP_MS / 1000.0)
        else:
            # Only reached when the loop was not broken: log the final partial
            # window, if there is one.
            if window:
                log_window(window, window_idx)

    if exit_code == EXIT_OK:
        print("\nStream complete. Open the MLflow UI and read the metric history.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
