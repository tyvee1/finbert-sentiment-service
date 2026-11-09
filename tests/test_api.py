"""
Integration tests for the Sentiment Analysis API.

Every test here maps to a stated submission requirement. The mapping is written
down so that a reader can tell required coverage from padding:

    test_predict_returns_valid_response ....... /predict schema + valid label
    test_predict_batch ........................ batch length matches input
    test_predict_batch_empty_list_returns_422 . batch 422 on empty list
    test_health_returns_ok_when_model_loaded .. /health 200 {"status": "ok"}
    test_predict_empty_text_returns_422 ....... /predict 422 on empty text
    test_health_returns_503_when_model_unloaded /health 503 otherwise
    test_predict_returns_503_when_unloaded .... /predict 503 unloaded
    test_predict_batch_returns_503_when_unloaded  batch 503 unloaded
    test_metrics_endpoint_exposes_prometheus .. /metrics via generate_latest
    test_request_counter_increments_per_prediction  counter per prediction
    test_error_counter_increments_on_failure .. error counter on exception
    test_log_entries_are_json_with_required_fields  JSON log shape

The first three are the three the specification names explicitly. The rest
exist because the specification also states behaviours — 503 on an unloaded
model, metrics incremented per prediction, the error counter on an exception —
that are otherwise verifiable only by reading the source. A guard that has only
ever been read has not been tested; four guards in this course's own lessons
fail open for exactly that reason.

The unloaded-model tests are the reason for the ``unloaded_model`` fixture: it
empties the module-level registry and restores it afterwards, so these tests
exercise the real production code path (``_require_model``) rather than a
parallel one written to be testable.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from app import main

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

HEADLINES = [
    "The company reported record profits and raised its dividend.",
    "The firm filed for bankruptcy after massive losses.",
    "Stocks closed flat on Friday amid low trading volume.",
]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def unloaded_model():
    """Temporarily empty the model registry, then restore it.

    Exercises the same ``_require_model()`` path production uses, instead of a
    separate "simulate unloaded" branch. Restoration happens even if the test
    fails, so one unloaded test cannot poison the session-scoped client that
    every other test shares.
    """
    saved = dict(main.classifiers)
    main.classifiers.clear()
    try:
        yield
    finally:
        main.classifiers.update(saved)


def _assert_prediction_shape(payload: dict) -> None:
    """Assert one response object carries the full documented result schema."""
    assert set(payload) >= {"text", "sentiment", "confidence", "latency_ms"}
    assert isinstance(payload["text"], str)
    assert payload["sentiment"] in VALID_SENTIMENTS
    assert isinstance(payload["confidence"], float)
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["latency_ms"], float)
    assert payload["latency_ms"] > 0


def _counter_total(metrics_text: str, metric_name: str, **labels) -> float:
    """Sum the samples of one counter from scraped exposition text.

    Reads the scraped output with the Prometheus parser rather than with a
    regex, so the assertion is against the exposition contract and does not
    depend on sample ordering or formatting.

    Args:
        metrics_text: Raw body of a ``GET /metrics`` response.
        metric_name: Counter name including its ``_total`` suffix.
        **labels: Optional label filter; samples must match all of them.

    Returns:
        The summed value, or 0.0 when the counter has no matching samples yet.
    """
    total = 0.0
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name != metric_name:
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                total += sample.value
    return total


# ---------------------------------------------------------------------------
# The three tests the specification names
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", HEADLINES)
def test_predict_returns_valid_response(client: TestClient, text):
    """POST /predict returns the full result schema and a valid sentiment."""
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    payload = response.json()
    _assert_prediction_shape(payload)
    assert payload["text"] == text


def test_predict_batch(client: TestClient):
    """POST /predict/batch returns one correctly shaped result per input."""
    response = client.post("/predict/batch", json={"texts": HEADLINES})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == len(HEADLINES)
    for item, text in zip(payload, HEADLINES, strict=True):
        _assert_prediction_shape(item)
        # Order is part of the contract: result i belongs to input i.
        assert item["text"] == text


def test_predict_batch_empty_list_returns_422(client: TestClient):
    """POST /predict/batch rejects an empty list with 422."""
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# The remaining stated behaviours
# ---------------------------------------------------------------------------
def test_health_returns_ok_when_model_loaded(client: TestClient):
    """GET /health returns exactly {"status": "ok"} with HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_empty_text_returns_422(client: TestClient):
    """POST /predict rejects empty text with 422."""
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_whitespace_only_text_returns_422(client: TestClient):
    """Whitespace-only text is treated as empty, not as valid input."""
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_health_returns_503_when_model_unloaded(client: TestClient, unloaded_model):
    """GET /health returns 503, not 200, when the model is absent.

    Asserted on the status code. A health endpoint written with Flask's
    ``return body, 503`` tuple syntax returns 200 in FastAPI with the tuple as
    the body, and would pass any check that merely looked for "503" in the
    source.
    """
    assert client.get("/health").status_code == 503


def test_predict_returns_503_when_unloaded(client: TestClient, unloaded_model):
    """POST /predict returns 503 when the model is absent."""
    response = client.post("/predict", json={"text": HEADLINES[0]})
    assert response.status_code == 503


def test_predict_batch_returns_503_when_unloaded(client: TestClient, unloaded_model):
    """POST /predict/batch returns 503 when the model is absent."""
    response = client.post("/predict/batch", json={"texts": HEADLINES})
    assert response.status_code == 503


def test_unloaded_model_takes_precedence_over_empty_input(
    client: TestClient, unloaded_model
):
    """With no model AND empty input, the service answers 503 rather than 422.

    The specification requires both codes but does not order them. Pinning the
    documented precedence here means a later refactor that moves validation
    into a Pydantic field constraint — which would silently invert it — fails a
    test instead of changing behaviour quietly.
    """
    assert client.post("/predict", json={"text": ""}).status_code == 503
    assert client.post("/predict/batch", json={"texts": []}).status_code == 503


# ---------------------------------------------------------------------------
# Metrics and logging
# ---------------------------------------------------------------------------
def test_metrics_endpoint_exposes_prometheus(client: TestClient):
    """GET /metrics returns exposition-format metrics with the right media type."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=" in response.headers["content-type"]
    body = response.text
    for name in (
        "prediction_requests_total",
        "prediction_latency_ms",
        "prediction_errors_total",
    ):
        assert name in body


def test_request_counter_increments_per_prediction(client: TestClient):
    """The request counter moves by the batch SIZE, not by one per request.

    Scrapes before and after and diffs, so the assertion is on observed metric
    values rather than on the presence of an ``.inc()`` call in the source.
    """
    before = _counter_total(client.get("/metrics").text, "prediction_requests_total")
    response = client.post("/predict/batch", json={"texts": HEADLINES})
    assert response.status_code == 200
    after = _counter_total(client.get("/metrics").text, "prediction_requests_total")
    assert after - before == pytest.approx(len(HEADLINES))


def test_latency_histogram_observes_each_prediction(client: TestClient):
    """The latency histogram's observation count moves with predictions made."""
    before = _counter_total(client.get("/metrics").text, "prediction_latency_ms_count")
    client.post("/predict", json={"text": HEADLINES[0]})
    after = _counter_total(client.get("/metrics").text, "prediction_latency_ms_count")
    assert after - before == pytest.approx(1.0)


def test_error_counter_increments_on_failure(client: TestClient):
    """The error counter moves when inference raises, and the error propagates.

    The failure is injected by replacing the loaded classifier with one that
    raises, so the exception travels the production path through
    ``run_predictions``' except block.
    """

    class FailingClassifier:
        """Stand-in classifier that always raises, to exercise the error arm."""

        def __call__(self, *args, **kwargs):
            raise RuntimeError("injected inference failure")

    saved = main.classifiers.get("sentiment")
    main.classifiers["sentiment"] = FailingClassifier()
    try:
        before = _counter_total(client.get("/metrics").text, "prediction_errors_total")
        with pytest.raises(RuntimeError, match="injected inference failure"):
            client.post("/predict", json={"text": HEADLINES[0]})
        after = _counter_total(client.get("/metrics").text, "prediction_errors_total")
        assert after - before == pytest.approx(1.0)
    finally:
        if saved is not None:
            main.classifiers["sentiment"] = saved
        else:  # pragma: no cover — only if the session fixture never loaded
            main.classifiers.pop("sentiment", None)


def test_log_entries_are_json_with_required_fields(client: TestClient, caplog):
    """Every emitted log line parses as JSON and carries the required fields.

    Asserts on captured output, not on the presence of ``json.dumps`` in the
    source: a helper that builds a dict and logs ``str(dict)`` looks correct and
    emits single-quoted Python repr, which no JSON parser accepts.
    """
    with caplog.at_level(logging.INFO, logger="finbert-api"):
        response = client.post("/predict", json={"text": HEADLINES[0]})
    assert response.status_code == 200

    records = [r.getMessage() for r in caplog.records if r.name == "finbert-api"]
    assert records, "no log entries were emitted for a prediction"
    for line in records:
        entry = json.loads(line)
        assert {"timestamp", "level", "message"} <= set(entry)

    prediction_entries = [
        json.loads(line) for line in records if json.loads(line)["message"] == "prediction"
    ]
    assert prediction_entries, "no 'prediction' log entry was emitted"
    assert {"sentiment", "confidence", "latency_ms"} <= set(prediction_entries[0])
