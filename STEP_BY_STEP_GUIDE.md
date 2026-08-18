# Step-by-Step Guide: From Notebook to Web API

This guide assumes you know Python and pandas/sklearn well (clearly
true, given the notebooks), but have never built a web API before. It
explains FastAPI from first principles, then walks through this
specific project file by file, then gives you the full workflow to run
it and grow it.

---

## Part 1 — What even *is* a web API, mechanically?

A web API is just a program that:
1. Sits and waits for HTTP requests to arrive over the network (a
   request is basically a text message like `POST /api/predict` plus a
   JSON body).
2. Runs some Python code in response.
3. Sends a text response back (usually JSON).

Two separate pieces make this happen:

- **A web framework** (FastAPI) — the library that lets you write
  `@app.get("/health")` above a function and have that function run
  when someone requests `/health`. It does NOT talk to the network
  itself.
- **A server** (uvicorn) — the actual program that opens a network
  socket, listens for real HTTP traffic, and hands each request to your
  FastAPI app. FastAPI without a server can't be reached by anyone;
  uvicorn without an app has nothing to run.

That's why you always start this project with:
```bash
uvicorn app.main:app --reload
```
"Take the object named `app` inside `app/main.py`, and serve it."

---

## Part 2 — The four concepts that explain 90% of the code

### 2.1 Routes are just decorated functions

```python
@app.get("/health")
async def health_check():
    return {"status": "ok"}
```
`@app.get("/health")` registers this function to run whenever a
`GET /health` request arrives. Whatever the function `return`s gets
converted to JSON automatically and sent back. There's no manual
"build a response object" step — you just return a Python dict.

`@app.post(...)` is the same idea for `POST` requests (used when the
client is *sending* data, like a customer record to predict on).

### 2.2 Type hints double as validation

This is FastAPI's signature feature. Compare:

```python
@app.post("/api/predict")
async def predict(customer: CustomerInput):
    ...
```

That one type hint, `customer: CustomerInput`, tells FastAPI: "parse
the incoming JSON body, check every field against the `CustomerInput`
class in `app/schemas.py` (right types? required fields present?
values in range?), and if anything's wrong, reject the request with a
detailed error automatically — before a single line of `predict()`
runs." You never write `if "Income" not in request_body: return error`
by hand.

`CustomerInput` itself is a `pydantic.BaseModel` — a class where each
attribute becomes a validated field:

```python
class CustomerInput(BaseModel):
    Income: float = Field(..., ge=0)   # required, must be >= 0
```

### 2.3 `async def` — don't worry about this much yet

`async def` lets a function pause while waiting on something slow (a
database call, a network request) so the server can handle *other*
requests in the meantime, instead of freezing. Our functions are fast
enough that this barely matters here, but writing `async def` is the
default modern style, so we use it throughout.

### 2.4 Startup logic runs once, not per-request

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()   # runs ONCE, when the server boots
    yield
    # (shutdown cleanup would go here, after `yield`)
```
This matters because loading a model file from disk takes real time.
Doing it inside `predict()` would reload the model on every single
request — brutally slow. Loading it once at startup and reusing it is
the standard pattern for any ML-serving API.

---

## Part 3 — Walking through this project's request flow

Follow one prediction request through the code, in order:

1. **Browser or client sends `POST /api/predict`** with a JSON body
   (see `app/schemas.py`'s example for the shape).

2. **`app/main.py` → `predict()`** receives it. FastAPI has already
   validated it against `CustomerInput` by the time this function body
   runs.

3. **`app/model_service.py` → `ModelService.predict()`** is called.
   It:
   - Converts the validated Pydantic object into a one-row pandas
     DataFrame.
   - Calls `feature_engineer.transform(...)` — this applies the exact
     same cleaning/feature-engineering steps as training (age
     calculation, education encoding, outlier capping, etc.) using
     **`common/feature_engineering.py`**, the single file both training
     and serving import from.
   - Calls `model.predict(...)` and `model.predict_proba(...)` on the
     loaded CatBoost model to get a cluster number and probabilities.
   - Looks up a human-readable name ("High-Value") from
     `models/segment_profiles.json`.

4. **Back in `main.py`**, the returned dict is validated against
   `PredictionOutput` and sent back as JSON.

5. **If you're using the web form (`app/templates/index.html`)**,
   JavaScript in that page's `<script>` tag does step 1 for you via
   `fetch('/api/predict', {...})`, then renders the JSON response into
   the result panel — no page reload needed.

---

## Part 4 — The full workflow, step by step

### Step 1 — Train the model
```bash
python train/train_pipeline.py
```
Reads `data/marketing_campaign.csv`, reproduces your notebooks' feature
engineering (with the `Education`/`Marital_Status` bug fixed) and
clustering, trains a tuned CatBoost classifier, and writes three files
into `models/`. **You must do this before the API can start** — it has
nothing to load otherwise.

### Step 2 — Run the API locally
```bash
uvicorn app.main:app --reload
```
`--reload` restarts the server automatically whenever you save a code
change — use this in development only, never in production (see
`DEPLOYMENT.md`).

Open:
- `http://127.0.0.1:8000/` — the web form
- `http://127.0.0.1:8000/docs` — interactive API docs, auto-generated.
  You can literally click "Try it out" and send a real request from
  your browser, no `curl` needed.

### Step 3 — Test your changes before trusting them
```bash
pytest tests/ -v
```
`tests/test_api.py` spins up the app in-process (no real server needed)
and checks that health, prediction, and validation all behave
correctly. Run this after any change to `app/` or `common/`.

### Step 4 — Package it
```bash
docker compose up --build
```
See `DEPLOYMENT.md` for what this does and why.

### Step 5 — Extend it (a worked example)

Say you want to add a new input field, `HasNewsletterSubscription`.

1. Add it to `CustomerInput` in `app/schemas.py`.
2. If it needs any transformation, add that logic to
   `_engineer()` in `common/feature_engineering.py` and add the
   resulting column name to `FEATURE_ORDER`.
3. Add the field to the web form in `app/templates/index.html`
   (and to the JS payload-building code in the same file).
4. Re-run `python train/train_pipeline.py` (the model needs to be
   retrained on data that includes the new feature).
5. Add a test case in `tests/test_api.py` covering the new field.
6. Run `pytest tests/ -v` and `ruff check app/ common/ train/ tests/`.

Because `common/feature_engineering.py` is imported by both training
and serving, you only ever write the transformation logic once — there
is no second copy to forget to update.

---

## Part 5 — Quick reference: FastAPI vs. what you might have half-seen elsewhere

| Concept | FastAPI |
|---|---|
| Define a URL handler | `@app.get("/path")` / `@app.post("/path")` decorator above a function |
| Read request data | Type-hint a Pydantic model as a parameter — validation is automatic |
| Return JSON | Just `return` a dict, dataclass, or Pydantic model |
| Return HTML | `Jinja2Templates` + `TemplateResponse` (see `app/main.py`'s `home()`) |
| Run the server | A separate program, `uvicorn`, not built into the framework |
| API docs | Free, auto-generated at `/docs` from your type hints — nothing to write |
