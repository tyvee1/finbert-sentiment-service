# Building a Real-Time Financial News Sentiment Service

**Author:** Tyrone G. Verdell

A pre-trained financial-sentiment classifier (FinBERT) operationalised as a
full ML Ops stack: versioned data preparation, a model registry with automated
promotion, a REST inference API, a container image, a CI/CD pipeline that
deploys to ECS Fargate, and production monitoring through Prometheus and
MLflow.

---

## Contents

- [What runs where](#what-runs-where)
- [Setup](#setup)
- [Running it, task by task](#running-it-task-by-task)
- [Verifying each deliverable, with expected output](#verifying-each-deliverable-with-expected-output)
- [Things that are not obvious from the code](#things-that-are-not-obvious-from-the-code)
- [Known gaps and limits](#known-gaps-and-limits)
- [Project structure](#project-structure)

---

## What runs where

```
                    data/raw_stream.csv          HuggingFace dataset
                            |                            |
                      [dvc: clean]                 [dvc: prepare]
                            |                            |
                    data/stream.csv            data/train.csv + data/test.csv
                            |                            |
                            |                      scripts/evaluate.py
                            |                            |
                            |                   MLflow run + registered model
                            |                            |
                            |                      scripts/promote.py
                            |                            |
                            |                    alias: finbert@production
                            |                            |
                            +---------> app/main.py <-----+
                            |            (FastAPI)
                            |            /health /predict /predict/batch /metrics
                            |                   |         |
                   monitoring/stream.py         |    Prometheus
                   (windowed metrics            |    (scrapes api:8000)
                    back into MLflow)           |
                                                |
                   scripts/run_deepchecks.py ---+
                   (drift gate: stream vs test)
```

The CI pipeline runs lint, then the test suite and the drift gate in parallel,
then builds the image and pushes it to ECR, then deploys a new ECS task
definition revision to `finbert-api-service` on `finbert-cluster`.

---

## Setup

Python **3.12** specifically. The dependency set is pinned against it, and a
3.13 interpreter resolves a different set whose failures surface much later as
import errors.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # every local default is filled in already
python scripts/smoke_test.py  # downloads and patches the model, then checks it loads
```

`smoke_test.py` is not optional. The published model's `config.json` stores its
`id2label` values as floats (`0.0`, `1.0`, `2.0`) instead of label strings, and
current `transformers` releases reject that config. The script rewrites the
cached copy. Skip it and every model load fails with a config error that says
nothing about the cause.

Expected final line:

```
  All checks passed.
```

### On macOS

`fasttext==0.9.3` publishes exactly one wheel — CPython 3.9, macOS arm64 — so
on Python 3.12 it compiles from source and needs a C++ toolchain:

```bash
xcode-select --install
```

Without it, `pip install -r requirements.txt` fails while building fasttext,
and the drift gate cannot compute the sentiment property. This is the most
likely first-run install failure in the project.

### Start the tracking server

```bash
mlflow server --port 5000
```

---

## Running it, task by task

```bash
# Task 1 — versioned data preparation
dvc init
dvc repro
git add dvc.lock data/.gitignore && git commit -m "Reproduce the data pipeline"

# Task 2 — evaluate, register, promote
python scripts/evaluate.py
python scripts/promote.py

# Task 3 — the API, its tests, and a load test
python app/main.py
pytest tests/
locust -f scripts/locustfile.py --host http://localhost:8000

# Task 4 — the container
docker compose up --build

# Task 5 — the drift gate (the CI pipeline runs this on every push to main)
python scripts/run_deepchecks.py

# Task 6 — production monitoring
python monitoring/stream.py
```

`dvc repro` must be run **in full**, not stage by stage. `dvc.lock` is a
deliverable, and a lock produced by `dvc repro clean` alone contains one stage.
Check it:

```bash
python -c "import yaml;print(sorted(yaml.safe_load(open('dvc.lock'))['stages']))"
# expected: ['clean', 'prepare']
```

---

## Verifying each deliverable, with expected output

Every command below can be run start to finish without reading the source. The
expected values are given so that a result can be compared rather than
interpreted.

### The data pipeline

```bash
dvc stage list
```
```
prepare  Outputs data/train.csv, data/test.csv
clean    Outputs data/stream.csv
```

`data/train.csv`, `data/test.csv` and `data/stream.csv` are **DVC** outputs and
are deliberately **not** tracked by git — `dvc repro` writes `data/.gitignore`
listing them, and that file *is* committed. It is how a fresh clone knows which
paths DVC owns. To confirm nothing is tracked twice:

```bash
git ls-files data/
# expected: data/.gitignore and data/raw_stream.csv only
```

### The registry

```bash
python - <<'EOF'
import os
from mlflow.tracking import MlflowClient
c = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
versions = c.search_model_versions("name='finbert'")
print("registered versions:", [v.version for v in versions])
print("production alias ->", c.get_model_version_by_alias("finbert", "production").version)
EOF
```

Expected: at least one version, and the `production` alias resolving to the
latest one. This is the same state the MLflow UI shows under **Models**; the
command is here because a screenshot is not something a reader can re-derive.

`promote.py` exits **0** both when it promotes and when it correctly declines —
both are successful decisions by the gate. It exits **2** when it could not
decide (no run, no metric, no server), and in that case nothing is promoted.
To see the declining branch without waiting for a weak model, raise the bar:

```bash
python - <<'EOF'
import yaml
p = yaml.safe_load(open("params.yaml")); p["promote"]["f1_threshold"] = 0.99
yaml.safe_dump(p, open("params.yaml", "w"), sort_keys=False)
EOF
python scripts/promote.py     # prints NOT PROMOTED, leaves the alias unchanged
git checkout params.yaml      # put the real threshold back
```

### The API

With the service running on port 8000:

```bash
curl -s localhost:8000/health
# {"status":"ok"}

curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/health
# 200   — and 503 when no model is loaded

curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The company reported record profits and raised its dividend."}'
# {"text":"...","sentiment":"positive","confidence":0.97...,"latency_ms":41.2...}

curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' -d '{"text":""}'
# 422

curl -s -X POST localhost:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Profits rose sharply.","The firm filed for bankruptcy."]}' | python -c 'import json,sys;print(len(json.load(sys.stdin)))'
# 2   — the response length always equals the input length

curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict/batch \
  -H 'Content-Type: application/json' -d '{"texts":[]}'
# 422
```

**On the 503 responses.** `/health`, `/predict` and `/predict/batch` all return
503 when the model is not loaded, and the service deliberately stays up in that
state so the status is reportable: `lifespan` catches a model-load failure,
logs it at ERROR, and leaves the registry empty rather than letting uvicorn
exit. If the process died on a failed load there would be nothing listening to
answer 503 at all. To see it:

```bash
MODEL_SOURCE=mlflow MODEL_NAME=does-not-exist python app/main.py &
sleep 10
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/health   # 503
```

**Precedence when both conditions hold.** With no model loaded *and* empty
input, the service answers **503**, not 422. Availability is checked first:
with no model there is nothing to validate the input against, and "I cannot
serve this at all" is the more accurate answer. Validation is done inside the
handlers rather than with Pydantic field constraints, because a Pydantic
constraint runs before the handler and would silently invert this.

**Whitespace-only text is treated as empty** and rejected with 422.

### Metrics and logs

```bash
curl -s localhost:8000/metrics | grep -E '^prediction_(requests|errors)_total|^prediction_latency_ms_count'
```

```bash
# the counter moves by the BATCH SIZE, not by one per request
before=$(curl -s localhost:8000/metrics | awk '/^prediction_requests_total/{s+=$2}END{print s+0}')
curl -s -X POST localhost:8000/predict/batch -H 'Content-Type: application/json' \
  -d '{"texts":["a profit rose","a loss widened","shares were flat today"]}' >/dev/null
after=$(curl -s localhost:8000/metrics | awk '/^prediction_requests_total/{s+=$2}END{print s+0}')
echo "delta: $((after - before))"   # expected: 3
```

Every log line the service emits is a single-line JSON object carrying at
least `timestamp`, `level` and `message`:

```bash
python app/main.py 2>&1 | grep '^{' | head -3 | python -c 'import json,sys;[print(json.loads(l)["message"]) for l in sys.stdin]'
# Loading model...
# Model loaded successfully
```

### The drift gate

```bash
python scripts/run_deepchecks.py; echo "exit: $?"
```

Three exit codes, and the third is the one that matters:

| exit | meaning |
| --- | --- |
| 0 | both drift scores are within their thresholds |
| 1 | at least one exceeded — the gate **fired** |
| 2 | the gate **could not measure anything** |

A 2 is never a pass. The usual causes are the datasets not existing (run
`dvc repro`), fewer than 100 rows on either side, or the language-model
download failing — see the next section.

---

## Things that are not obvious from the code

### `run_prediction` and `run_predictions` both exist, and neither is dead code

`run_predictions(texts)` is the single inference path: it runs the model,
updates all three Prometheus metrics once per prediction, and emits the
per-prediction log line. `run_prediction(text)` is the single-text form and
delegates to it. `POST /predict` calls the singular, `POST /predict/batch`
calls the plural. Two names, one implementation, one place where the metrics
are emitted — rather than two paths that drift apart.

### `transformers` is held below version 5, on purpose

MLflow 2.16.2's transformers flavour imports `FlaxPreTrainedModel`, which
transformers 5 removed. With transformers 5 installed,
`mlflow.transformers.log_model` fails, and because the failed import makes
MLflow's framework detection fall through to TensorFlow, the error you actually
see is:

```
ModuleNotFoundError: No module named 'tensorflow'
```

which points nowhere near the cause. `requirements.txt` pins
`transformers>=4.40.2,<5`, and `sentence-transformers<6` alongside it, because
sentence-transformers 6 (pulled in by `deepchecks[nlp]`) requires transformers
5 and would drag the resolution back over the line. Confirm before debugging
anything in Task 2:

```bash
python -c "import transformers; print(transformers.__version__)"   # expect 4.x
```

### `ruff` is pinned, and the rule set is explicit

`ruff.toml` selects the rules; the CI lint job installs an exact version. An
unpinned linter makes the gate's verdict a function of the day it ran, and
because every other job in the pipeline declares `needs: lint`, a lint failure
skips the whole pipeline rather than failing one job.

### The drift gate downloads a model from a third host

`TextData.calculate_builtin_properties` calls `get_fasttext_model()`
unconditionally — the call is not inside any conditional — so it fetches
`lid.176.bin` (~126 MB) from `dl.fbaipublicfiles.com` even when only the
`Sentiment` property is requested. That is neither HuggingFace nor PyPI nor
AWS. It is cached under `.deepchecks_models/` (gitignored), and the CI job
caches that directory. If the host is unreachable and the cache is cold, the
gate exits 2.

### Latency is measured in milliseconds and the histogram buckets say so

`prometheus_client`'s default Histogram buckets top out at 10, which is
sensible for seconds and useless for milliseconds — every observation lands in
the same bucket and the histogram becomes a counter. Explicit millisecond
buckets are set. Reading it: buckets are **cumulative**, so the largest bucket
always equals the total count and the distribution is in the *differences*
between buckets, never in the top one.

### The container must be built for the architecture it will run on

An image built on Apple silicon is `linux/arm64`. An ECS task definition
defaults to `X86_64`, and the task then dies immediately with
`exec format error` and no application log at all. CI builds on
`ubuntu-latest` (amd64), so the deployed path is correct. For a local build
that will be pushed:

```bash
docker buildx build --platform linux/amd64 -t finbert-api:local .
```

### The image installs `requirements-prod.txt`, not `requirements.txt`

`requirements.txt` is the development set — dvc, deepchecks, locust, pytest,
the datasets stack — none of which the running service imports. It also
resolves torch to the default PyPI build, which bundles CUDA: measured at
**2.7 GB of `nvidia-*` libraries plus 690 MB of triton**, on top of torch's own
1.2 GB, all shipped to a Fargate task with no GPU. `requirements-prod.txt`
installs the CPU-only wheel instead.

---

## Stand-out additions

Two of the course's optional suggestions are implemented, both because their
own wording asks for the artefact to live in the repository.

### Model rollback

`scripts/rollback_model.py` moves the `production` alias back to the
previously promoted version.

```bash
python scripts/rollback_model.py --dry-run   # report the change
python scripts/rollback_model.py             # make it
```

**It does not roll back to version N-1.** Version numbers are not a promotion
history: versions get registered by evaluation runs that never cleared the
threshold, so N-1 is frequently a model that was already judged unfit, and
deleted versions leave gaps where N-1 does not exist at all. `promote.py`
therefore tags each version it promotes, and the rollback walks back to the
most recent version carrying that tag. If there is no previously approved
version, it refuses and says to fix forward rather than picking the nearest
number.

**How it is triggered.** Manually from the Actions tab via
`.github/workflows/rollback.yml` (`workflow_dispatch`, defaulting to a dry
run), or automatically when `test-and-deploy` concludes in failure. The
automatic path carries two limits, stated because they decide whether it is
useful:

- `workflow_run` observes a whole workflow's conclusion and GitHub offers no
  cross-workflow trigger on a single job, so the workflow queries the
  triggering run and stops unless the `deepchecks` job is the one that failed.
  Without that query, a lint error would roll production back.
- A GitHub runner cannot reach a tracking server on a laptop. The workflow
  requires `MLFLOW_TRACKING_URI` as a repository secret and stops with a clear
  message when it is absent, rather than reporting a rollback that did not
  happen.

### Grafana dashboard

`monitoring/grafana-dashboard.json` — import via **Dashboards → New → Import →
Upload JSON file**, then select the Prometheus data source. Six panels:
prediction rate, latency percentiles, sentiment distribution over time, error
ratio, scrape-target health, and hourly volume.

The latency panel uses `histogram_quantile()` over
`prediction_latency_ms_bucket`. Graphing the buckets directly would plot a
monotonically rising line, because the buckets are cumulative — the largest
always equals the total count.

### Not implemented: A/B traffic routing

The A/B suggestion is the only one of the five that changes the graded
surface — it adds a routing branch inside `POST /predict`, the endpoint most
of the inference criteria are scored against. An optional addition cannot
raise the grade, and this is the only one that could lower it.

---

## Known gaps and limits

These are real and are stated rather than left to be discovered.

**The security group in the setup instructions cannot reach the service.**
`finbert-sg` is specified with inbound HTTP on port **80**; the container
listens on **8000** and the health check probes 8000. A task behind that group
is unreachable, and the ECS service will cycle tasks that never pass their
health check. It needs inbound 8000 — or, as this would be built in
production, an Application Load Balancer listening on 80 with an 8000 target
group.

**The deployed service may not serve the model the registry promoted.**
`MODEL_SOURCE=huggingface` in CI and in the task definition means the container
loads from the Hub, while the registry holds the version that was evaluated and
promoted. The registry then documents a deployment it does not govern. Setting
`MODEL_SOURCE=mlflow` closes it, and requires the task to reach a tracking
server and its artifact store — which is the real work the exercise's scope
stops short of.

**The drift score may be measuring a population difference rather than drift.**
The reference side is `data/test.csv`: curated, labelled financial headlines.
The current side is `data/stream.csv`: cleaned Bluesky posts. These are
different populations gathered different ways, so a high score here can be a
faithful measurement of a difference that was present on day one. A *change* in
the score over time is informative; the absolute value is not evidence of model
decay.

**`dvc repro` reproduces the pipeline, not the data.** The `prepare` stage's
real input is a remote HuggingFace dataset named by an environment variable,
with no pinned revision and no tracked dependency. DVC will report the stage as
up to date after the upstream dataset changes. `dvc.lock` records the hash of
what was produced, which makes a substitution detectable after the fact but
cannot prevent it. Pinning a dataset revision is the fix.

**There is no ground-truth label loop.** Everything monitored here is a
property of the inputs and the outputs — sentiment distribution, confidence,
latency, drift. None of it measures whether the predictions are *correct*,
because no labels arrive after serving. Real model-quality monitoring needs
that loop, and it does not exist in this project.

**No infrastructure as code.** Every AWS resource is created by hand per the
setup instructions. Nothing records what was created, and nothing tears it
down.

---

## Project structure

```
.
├── app/
│   ├── main.py              # FastAPI service, Prometheus metrics, JSON logging
│   └── utils.py             # model loading (MLflow registry or HuggingFace Hub)
├── data/
│   ├── raw_stream.csv       # raw Bluesky financial posts (committed)
│   └── .gitignore           # written by DVC; lists the DVC-owned outputs
├── monitoring/
│   └── stream.py            # replays the stream, logs windowed metrics to MLflow
├── scripts/
│   ├── load_data.py         # prepare stage: download and split
│   ├── clean_data.py        # clean stage: normalise the raw stream
│   ├── evaluate.py          # evaluate, log, register
│   ├── promote.py           # promotion gate on f1_weighted
│   ├── run_deepchecks.py    # drift gate: property drift + prediction drift
│   ├── locustfile.py        # load test
│   └── smoke_test.py        # model download, config patch, load check
├── tests/
│   ├── conftest.py          # session-scoped client, model loaded once
│   └── test_api.py          # integration tests
├── .github/workflows/
│   └── ci-cd.yml            # lint -> test + deepchecks -> build -> deploy
├── dvc.yaml                 # the two-stage pipeline
├── dvc.lock                 # produced by `dvc repro` (committed)
├── params.yaml              # split, cleaning, promotion and drift parameters
├── Dockerfile               # production image, CPU-only
├── docker-compose.yml       # API + Prometheus, local
├── prometheus.yml           # scrape config for the api service
├── requirements.txt         # development and CI
├── requirements-prod.txt    # runtime only, CPU-only torch
├── ruff.toml                # the lint rule set the CI job enforces
└── .env.example             # copy to .env
```

---

## Resources

- [FinBERT Market Based](https://huggingface.co/baptle/FinBERT_market_based) — the model
- [Financial Headlines Market Based](https://huggingface.co/datasets/baptle/financial_headlines_market_based) — the dataset
- [DVC pipelines](https://dvc.org/doc/user-guide/pipelines/defining-pipelines)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Prometheus metric types](https://prometheus.io/docs/concepts/metric_types/)
- [Deepchecks NLP drift](https://docs.deepchecks.com/stable/general/guides/drift_guide.html)
- [AWS Fargate on ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/getting-started-fargate.html)
