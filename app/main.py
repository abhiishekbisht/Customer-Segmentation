"""
app/main.py
============
This is the entry point of the whole web application.

CONCEPTS FOR SOMEONE NEW TO FASTAPI (skip if you already know these):
-----------------------------------------------------------------------
- An FastAPI "app" object is a single Python object that represents your
  entire web server. You attach "routes" to it (a route = a URL path +
  an HTTP method, like "GET /" or "POST /api/predict").

- You attach a route with a DECORATOR: `@app.get("/")`. The decorator
  wraps the function right below it and registers it as the handler for
  that URL. When a browser requests that URL, FastAPI calls your function
  and sends whatever it returns back as the HTTP response.

- `async def` vs `def`: FastAPI supports both. `async def` lets FastAPI
  run many requests concurrently without blocking on I/O (like reading
  a file or calling a database). We use `async def` throughout since
  it's the modern default; our functions are fast enough it doesn't
  matter much here, but it's good habit for real APIs.

- Type hints in a function's parameters (e.g. `customer: CustomerInput`)
  are not just documentation here -- FastAPI reads them at runtime to
  know HOW to parse and validate the incoming request. This is the
  single biggest difference from Flask, where you'd manually pull values
  out of `request.json` or `request.form` and validate them yourself.

Run this app with (from the project root):
    uvicorn app.main:app --reload

Breaking that command down:
    uvicorn            -> the ASGI server that actually runs the app
                           (FastAPI itself doesn't include a server)
    app.main:app        -> "in the app/main.py module, use the object
                           named `app`" (that's the FastAPI() instance below)
    --reload            -> restart the server automatically when you
                           edit code (development only -- never use in
                           production, see DEPLOYMENT.md)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model_service import model_service
from app.schemas import CustomerInput, PredictionOutput

# --------------------------------------------------------------------
# Logging: print structured, timestamped messages instead of using
# bare `print()`. Any real deployment platform (Docker, cloud logs,
# etc.) captures stdout, so logging.info() calls show up there.
# --------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("customer_segmentation_api")

# --------------------------------------------------------------------
# LIFESPAN: code that runs once at startup, and once at shutdown
# --------------------------------------------------------------------
# `@asynccontextmanager` turns this generator into something FastAPI can
# run around the app's lifetime: everything before `yield` runs at
# startup (we use it to load the ML model into memory ONE time -- see
# app/model_service.py for why that matters); everything after `yield`
# runs at shutdown (we don't need cleanup here, but that's where it'd go
# -- e.g. closing a database connection pool).
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts ...")
    model_service.load()
    logger.info("Model artifacts loaded. API is ready to serve predictions.")
    yield
    logger.info("Shutting down.")


# --------------------------------------------------------------------
# Create the FastAPI app object. Everything below attaches to `app`.
# The metadata here (title/description/version) auto-populates the
# interactive docs FastAPI generates for free at /docs and /redoc.
# --------------------------------------------------------------------
app = FastAPI(
    title="Customer Segmentation API",
    description="Predicts which customer segment (cluster) a customer belongs to.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve files in app/static/ (CSS, JS, images) at the URL prefix /static.
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2Templates lets a Python route return a rendered HTML page
# (app/templates/*.html) instead of raw JSON -- this is how we build
# the human-facing web UI on top of the same FastAPI app.
templates = Jinja2Templates(directory="app/templates")


# --------------------------------------------------------------------
# ROUTE 1: Health check
# --------------------------------------------------------------------
# Every production API should have a cheap endpoint that says "I'm
# alive and my dependencies are loaded". Load balancers, Docker, and
# Kubernetes all poll something like this to decide whether to send
# traffic to this instance.
@app.get("/health")
async def health_check():
    return {"status": "ok" if model_service.is_ready() else "not_ready"}


# --------------------------------------------------------------------
# ROUTE 2: The human-facing web page (HTML)
# --------------------------------------------------------------------
# `response_class=HTMLResponse` tells FastAPI (and its docs) that this
# route returns HTML, not JSON. `Request` is required by Jinja2Templates
# so it can build correct URLs inside the template.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


# --------------------------------------------------------------------
# ROUTE 3: The JSON prediction API
# --------------------------------------------------------------------
# This is the "real" API endpoint -- the one other programs (our own
# front-end JavaScript, a mobile app, another backend service, Postman,
# curl, etc.) call directly.
#
# `customer: CustomerInput` in the signature means: "parse the request
# body as JSON, validate it against the CustomerInput schema (see
# app/schemas.py), and if it's invalid, automatically return an HTTP 422
# error with details -- before a single line of this function runs."
#
# `response_model=PredictionOutput` means FastAPI will validate and
# document the shape of what we return, too.
@app.post("/api/predict", response_model=PredictionOutput)
async def predict(customer: CustomerInput):
    if not model_service.is_ready():
        # HTTPException is FastAPI's way of returning a proper HTTP
        # error status + JSON error body, instead of a generic crash.
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        result = model_service.predict(customer)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return result
