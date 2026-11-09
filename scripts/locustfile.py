"""
Load test for the Sentiment Analysis API.

Run:
    locust -f scripts/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure and start the test.

Task weights model the traffic mix a sentiment service actually sees: single
predictions dominate, batch calls are occasional, and health checks arrive at
roughly the cadence a load balancer or an orchestrator probes at.

    /predict        weight 3   single-headline scoring, the hot path
    /predict/batch  weight 1   bulk scoring
    /health         weight 1   liveness probing

`wait_time` is a range rather than a constant. A constant think time makes
every simulated user fire in lockstep, which produces a synthetic request
pattern — regular spikes with idle gaps — and a latency distribution that looks
better than production because the service is never asked to handle genuinely
concurrent arrivals.

A note on reading the results: with FinBERT on CPU, latency is dominated by
inference, not by the HTTP layer, so throughput saturates at roughly
(number of worker processes) requests in flight. Response times climbing
linearly with user count is queueing, not a slow model, and adding uvicorn
workers moves that ceiling where tuning the API will not.
"""

import random

from locust import HttpUser, between, task

SAMPLE_HEADLINES = [
    "The company reported record profits and raised its dividend.",
    "The firm filed for bankruptcy after massive losses.",
    "Stocks closed flat on Friday amid low trading volume.",
    "Federal Reserve signals interest rate cuts later this year.",
    "Tech giant announces major layoffs amid revenue decline.",
    "Merger talks between the two firms collapsed overnight.",
    "Quarterly earnings beat analyst expectations by wide margin.",
    "Oil prices surge on supply concerns from the Middle East.",
]

BATCH_SIZE = 4


class SentimentAPIUser(HttpUser):
    """One simulated client of the sentiment API."""

    # Uniform think time between requests, in seconds.
    wait_time = between(0.5, 2.0)

    @task(3)
    def predict_single(self) -> None:
        """POST one randomly chosen headline to /predict.

        ``name`` is set explicitly so every request aggregates under one entry
        in the Locust statistics table. Without it, URLs that differ per
        request would each get their own row and the percentile columns would
        be computed over samples of one.
        """
        self.client.post(
            "/predict",
            json={"text": random.choice(SAMPLE_HEADLINES)},
            name="/predict",
        )

    @task(1)
    def predict_batch(self) -> None:
        """POST a random sample of headlines to /predict/batch."""
        self.client.post(
            "/predict/batch",
            json={"texts": random.sample(SAMPLE_HEADLINES, BATCH_SIZE)},
            name="/predict/batch",
        )

    @task(1)
    def health_check(self) -> None:
        """GET /health.

        A 503 here is a correct response from a service with no model loaded,
        not a load-test failure, so it is reported separately rather than
        counted as an error. ``catch_response`` is what makes that distinction
        possible: without it Locust records any non-2xx as a failure and a
        run against a model-less service reports a 100% error rate that says
        nothing about the service's behaviour under load.
        """
        with self.client.get(
            "/health", name="/health", catch_response=True
        ) as response:
            if response.status_code == 503:
                response.failure("service reports model not loaded (503)")
            elif response.status_code != 200:
                response.failure(f"unexpected status {response.status_code}")
            else:
                response.success()
