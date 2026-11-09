"""
Sentiment Prediction API.

A FastAPI service that wraps a FinBERT financial-sentiment classifier and
exposes it over HTTP, instrumented for Prometheus scraping and structured
(JSON) logging.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics (text exposition format)

--------------------------------------------------------------------------
Design notes a reader should not have to reverse-engineer
--------------------------------------------------------------------------

1.  MODEL LOAD FAILURE IS CAUGHT, ON PURPOSE.

    ``lifespan`` wraps ``load_classifier()`` in try/except. That looks like a
    swallowed error and it is the opposite. If the load raised, uvicorn would
    exit during startup and there would be no process left to answer
    ``GET /health`` with 503 — the very state the specification requires the
    service to be able to report. Catching it leaves the service up, the
    ``classifiers`` registry empty, and every prediction path returning 503.

    The exception is logged at ERROR with its type and message, so a caught
    failure is loud, not silent. This is fail-CLOSED for predictions (nothing
    is served) while keeping the health signal observable.

2.  503 IS RETURNED BY RAISING, NOT BY RETURNING A TUPLE.

    ``return {...}, 503`` is Flask. In FastAPI it returns a 200 whose body is a
    two-element array, so a health endpoint written that way reports healthy
    while the model is missing. Every non-200 here goes through
    ``HTTPException``, and the tests assert on ``response.status_code`` rather
    than on the presence of "503" anywhere in this file.

3.  PRECEDENCE: 503 BEFORE 422.

    The specification requires 422 for empty input and 503 for an unloaded
    model, and does not say which wins when both hold. This service checks
    availability first: with no model loaded there is nothing to validate
    input against, and 503 ("I cannot serve this at all") is the more honest
    answer than 422 ("your request was malformed"). Validation is therefore
    done inside the handlers rather than with Pydantic field constraints,
    because a Pydantic constraint runs before the handler and would silently
    invert this ordering. Both orderings are pinned by tests.

4.  WHITESPACE-ONLY TEXT COUNTS AS EMPTY.

    ``{"text": "   "}`` is rejected with 422. A string of spaces is not input a
    sentiment classifier can say anything meaningful about, and accepting it
    would produce a confident-looking prediction over nothing.

5.  LATENCY IS MEASURED IN MILLISECONDS, AND THE HISTOGRAM BUCKETS SAY SO.

    ``prometheus_client``'s default Histogram buckets top out at 10 — sensible
    for a metric measured in seconds, useless for one measured in milliseconds,
    where every observation lands in the same bucket and the histogram becomes
    a counter wearing a costume. Because buckets are CUMULATIVE, the largest
    bucket always equals the total count; reading a histogram means reading the
    DIFFERENCES between buckets, not the top one. Explicit millisecond buckets
    are set below.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

# The model loader is imported under both spellings it is reachable by.
#
# ``app/main.py`` is imported three different ways in this project and they do
# not agree on what the package root is:
#   * ``python app/main.py``        — app/ is sys.path[0]  -> ``utils``
#   * ``pytest`` via tests/conftest — repo root on sys.path -> ``app.utils``
#   * ``uvicorn app.main:app``      — repo root on sys.path -> ``app.utils``
# The starter shipped only the first spelling, which makes the container's
# ``uvicorn app.main:app`` entry point fail on import. Both are accepted here
# so the same file works under all three.
try:
    from app.utils import load_classifier
except ImportError:  # pragma: no cover — exercised by `python app/main.py`
    from utils import load_classifier

# ---------------------------------------------------------------------------
# Structured logging (Task 6)
# ---------------------------------------------------------------------------
# The handler format is the bare message: ``log()`` has already produced a
# complete JSON document, and any prefix logging adds ("INFO:root:...") would
# make the line unparseable to a log shipper. force=True so this configuration
# wins even when uvicorn has already installed its own handlers.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
)
_logger = logging.getLogger("finbert-api")


def log(level: str, message: str, **kwargs) -> None:
    """Emit one structured log entry as a single-line JSON object.

    Every entry carries at least ``timestamp``, ``level`` and ``message``; any
    keyword arguments are merged in as additional fields. One JSON document per
    line is what makes the stream machine-readable — a multi-line record cannot
    be split on newlines by a collector.

    Args:
        level: Severity name, e.g. ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
        message: Human-readable event description.
        **kwargs: Extra structured fields (``sentiment``, ``latency_ms``, ...).
            Values that are not JSON-serialisable are coerced with ``str`` via
            ``default=str`` rather than raising, because a logging call must
            never be the thing that breaks a request.

    Returns:
        None. The entry goes to the configured logging handler (stdout under
        both uvicorn and Docker).
    """
    entry = {
        # Timezone-aware UTC. A naive timestamp is ambiguous the moment the
        # service runs anywhere other than the machine that read the log.
        "timestamp": datetime.now(UTC).isoformat(),
        "level": level.upper(),
        "message": message,
    }
    entry.update(kwargs)
    line = json.dumps(entry, default=str)

    # Map the level name onto the logger so standard level filtering still
    # works; unknown names fall back to INFO rather than being dropped.
    _logger.log(logging.getLevelNamesMapping().get(level.upper(), logging.INFO), line)


# ---------------------------------------------------------------------------
# Prometheus metrics (Task 6)
# ---------------------------------------------------------------------------
# Naming follows the Prometheus convention: ``_total`` suffix on counters, unit
# suffix on everything measured. prometheus_client appends ``_total`` to a
# Counter itself, so the constructor is given the stem.
PREDICTION_REQUESTS = Counter(
    "prediction_requests",
    "Total number of predictions returned, labelled by predicted sentiment.",
    ["sentiment"],
)

# Buckets in MILLISECONDS, chosen for CPU FinBERT inference (tens of ms for a
# single short headline, hundreds for a batch). The top bucket is deliberately
# well above expected latency so saturation is visible rather than clipped.
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds, measured around model inference.",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)

PREDICTION_ERRORS = Counter(
    "prediction_errors",
    "Total number of prediction attempts that raised an exception.",
)

# Text longer than this is logged as a warning before inference. FinBERT
# truncates at 512 tokens silently; without this the caller gets a confident
# prediction over a prefix of their input and no indication it happened.
MAX_TEXT_CHARS = 2000

load_dotenv()

# Module-level registry rather than a global model variable: ``lifespan``
# populates it on startup and clears it on shutdown, and "is the model loaded?"
# becomes a single membership test that the health endpoint and both prediction
# endpoints share.
classifiers = {}


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    text: str


class PredictBatchRequest(BaseModel):
    """Request body for ``POST /predict/batch``."""

    texts: list[str]


class PredictionResult(BaseModel):
    """One prediction: the input echoed back, the label, its score, and timing.

    ``latency_ms`` is wall-clock milliseconds around the inference call. For a
    batch it is the per-item share of the batch's total, so the field means the
    same thing on both endpoints rather than being the whole batch's time
    repeated N times.
    """

    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier on startup and release it on shutdown.

    A load failure is caught and logged rather than propagated — see design
    note 1 in the module docstring. The service stays up with an empty
    ``classifiers`` registry so that ``/health`` can report 503.
    """
    log("INFO", "Loading model...", model_source=os.getenv("MODEL_SOURCE", "mlflow"))
    try:
        classifiers["sentiment"] = load_classifier()
        log("INFO", "Model loaded successfully")
    except Exception as exc:  # noqa: BLE001 - intentional, see design note 1
        # Deliberate: keep the process alive so /health is answerable.
        log(
            "ERROR",
            "Model failed to load; service will report unhealthy",
            error=type(exc).__name__,
            detail=str(exc),
        )
    yield
    classifiers.clear()
    log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def _require_model():
    """Return the loaded classifier or raise 503.

    One place decides what "the model is available" means, so the health
    endpoint and both prediction endpoints can never disagree about it.

    Raises:
        HTTPException: 503 when no classifier is loaded.
    """
    classifier = classifiers.get("sentiment")
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return classifier


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    """Run batch inference and return one ``PredictionResult`` per input text.

    This is the single inference path in the service: both ``/predict`` and
    ``/predict/batch`` reach the model through it, so the Prometheus metrics and
    the per-prediction log lines are emitted once, here, rather than being
    duplicated per endpoint and drifting apart.

    Metrics updated (Task 6):
        * ``PREDICTION_REQUESTS`` — incremented once PER PREDICTION, labelled
          by the predicted sentiment. A batch of 10 moves it by 10, not by 1.
        * ``PREDICTION_LATENCY`` — observed once per prediction, in ms.
        * ``PREDICTION_ERRORS`` — incremented once if inference raises, before
          the exception is re-raised.

    Args:
        texts: Input strings. Assumed already validated as non-empty by the
            caller; this function does not re-validate, so that there is one
            place where the 422 contract lives.

    Returns:
        A list of ``PredictionResult`` the same length and order as ``texts``.

    Raises:
        Exception: whatever the underlying model raised, re-raised unchanged
            after the error counter is incremented and the error is logged.
    """
    try:
        # Visibility for silent truncation: FinBERT caps at 512 tokens, so a
        # long input is scored on a prefix with no signal to the caller.
        for text in texts:
            if len(text) > MAX_TEXT_CHARS:
                log(
                    "WARNING",
                    "Input exceeds model context and will be truncated",
                    text_chars=len(text),
                    limit_chars=MAX_TEXT_CHARS,
                )

        classifier = classifiers["sentiment"]

        # perf_counter_ns rather than perf_counter: a nanosecond counter always
        # advances across a real call, so a measured latency is never exactly
        # 0.0 and "latency_ms > 0" stays a meaningful assertion rather than one
        # that holds only because the model is slow. perf_counter (not
        # time.time) because it is monotonic — a wall-clock adjustment mid-batch
        # can otherwise produce a negative duration.
        start_ns = time.perf_counter_ns()
        raw_results = classifier(texts)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

        # Per-item share of the batch's wall time, so latency_ms is comparable
        # across the single and batch endpoints.
        per_item_latency_ms = elapsed_ms / len(texts) if texts else elapsed_ms

        results: list[PredictionResult] = []
        # strict=True: a pipeline returning a different number of results than it
        # was given is a bug, and silently truncating to the shorter of the two
        # would hand the caller predictions belonging to other inputs.
        for text, raw in zip(texts, raw_results, strict=True):
            # The published checkpoint ships config.json with id2label stored as
            # FLOATS, so an unpatched cache makes the pipeline return
            # {"label": 2.0} instead of {"label": "positive"}. Left to Pydantic,
            # that surfaces as a ValidationError on a field the caller never
            # sent, the endpoint answers 500 with "Internal Server Error", and
            # /health still answers 200 because the model itself loaded fine —
            # three symptoms that point away from the actual cause. Naming the
            # cause here costs one isinstance check per prediction and is the
            # difference between a one-line fix and an afternoon.
            label = raw["label"]
            if not isinstance(label, str):
                raise RuntimeError(
                    "the model returned a non-string label "
                    f"({label!r}, type {type(label).__name__}); the cached "
                    "config.json has numeric id2label values and needs "
                    "patching — run `python scripts/smoke_test.py`"
                )

            result = PredictionResult(
                text=text,
                sentiment=label,
                confidence=float(raw["score"]),
                # 4 decimal places = 0.1 microsecond resolution. Enough that
                # any real measurement is non-zero, few enough to read.
                latency_ms=round(per_item_latency_ms, 4),
            )
            results.append(result)

            PREDICTION_REQUESTS.labels(sentiment=result.sentiment).inc()
            PREDICTION_LATENCY.observe(result.latency_ms)
            log(
                "INFO",
                "prediction",
                sentiment=result.sentiment,
                confidence=round(result.confidence, 6),
                latency_ms=result.latency_ms,
            )

        return results
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        log(
            "ERROR",
            "Prediction failed",
            error=type(exc).__name__,
            detail=str(exc),
            batch_size=len(texts),
        )
        raise


def run_prediction(text: str) -> PredictionResult:
    """Run inference on a single text and return its prediction and latency.

    The single-text form of :func:`run_predictions`, and the function
    ``POST /predict`` is implemented in terms of. It delegates rather than
    duplicating, so there is exactly one code path to the model and the metrics
    and logs are emitted in exactly one place.

    Both names exist because the specification refers to this helper in the
    singular ("a helper function, ``run_prediction``, that accepts a text") and
    in the plural ("the ``run_predictions`` function increments the request
    counter") in two different sections. Neither name is dead code: ``/predict``
    calls this one, ``/predict/batch`` calls the plural one.

    Args:
        text: A single non-empty input string.

    Returns:
        The ``PredictionResult`` for ``text``, carrying both the prediction
        (``sentiment``, ``confidence``) and the measured ``latency_ms``.
    """
    return run_predictions([text])[0]


@app.get("/metrics")
def metrics() -> Response:
    """Expose the Prometheus metrics in text exposition format.

    ``CONTENT_TYPE_LATEST`` is returned as the media type rather than a
    hard-coded ``text/plain``: Prometheus negotiates the exposition format by
    content type, and the constant tracks whatever version of it the installed
    client library produces.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict:
    """Report service health.

    Returns:
        ``{"status": "ok"}`` with HTTP 200 when the model is loaded.

    Raises:
        HTTPException: 503 when the model is not loaded. Raised, not returned
            as a tuple — see design note 2 in the module docstring.
    """
    _require_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest) -> PredictionResult:
    """Classify a single headline.

    Args:
        request: Body carrying ``text``.

    Returns:
        A ``PredictionResult`` for the input.

    Raises:
        HTTPException: 503 when the model is not loaded (checked first), or 422
            when ``text`` is empty or whitespace-only.
    """
    _require_model()
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Field 'text' must not be empty")
    return run_prediction(request.text)


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest) -> list[PredictionResult]:
    """Classify a list of headlines.

    Args:
        request: Body carrying ``texts``.

    Returns:
        A list of ``PredictionResult``, one per input, in input order.

    Raises:
        HTTPException: 503 when the model is not loaded (checked first), 422
            when ``texts`` is empty, or 422 when any element is empty or
            whitespace-only. The per-element check is not required by the
            specification; without it one blank string in a batch of fifty
            yields a confident prediction over nothing.
    """
    _require_model()
    if not request.texts:
        raise HTTPException(status_code=422, detail="Field 'texts' must not be empty")
    if any(not text.strip() for text in request.texts):
        raise HTTPException(
            status_code=422, detail="Field 'texts' must not contain empty strings"
        )
    return run_predictions(request.texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
    )
