# Deployment Guide — Getting This to Production the Right Way

This covers what's already built into the project, what you still need
to decide, and the reasoning behind each choice — so you're not just
copy-pasting config you don't understand.

## 1. What "production-ready" already means in this repo

| Practice | Where | Why it matters |
|---|---|---|
| Pinned dependency versions | `requirements.txt` | An unpinned install six months from now can silently pull a breaking change (this project itself hit a breaking Starlette API change during development). |
| Multi-stage Docker build | `Dockerfile` | Build tools (compilers, headers) needed to install packages don't end up in the final image — smaller, faster, fewer things that can have vulnerabilities. |
| Non-root container user | `Dockerfile` | If the app is ever compromised, the attacker doesn't get root inside the container. |
| Health check endpoint | `GET /health`, `Dockerfile HEALTHCHECK` | Load balancers and orchestrators need a cheap way to ask "are you actually working?", not just "is the process running?" |
| No `--reload` in the container CMD | `Dockerfile` | `--reload` watches the filesystem and re-imports code on every change — pure overhead and a footgun in production. |
| Input validation before business logic runs | `app/schemas.py` | Untrusted input never reaches the model without being checked first. |
| Automated tests + lint gating merges | `.github/workflows/ci.yml` | Nothing reaches `main` without passing tests, so regressions get caught before deployment, not after. |
| Structured logging | `app/main.py` (`logging` module) | `print()` statements are easy to lose; `logging` gives timestamps/levels and integrates with any log aggregator you point it at later. |
| Single source of truth for feature engineering | `common/feature_engineering.py` | Eliminates train/serve skew — the #1 cause of "worked in the notebook, broke in prod" bugs. |

## 2. What you still need to decide before going live

### 2.1 Where to host it
For a solo/small-team project like this, roughly in order of
simplicity:

| Option | Good for | Notes |
|---|---|---|
| **Render / Railway / Fly.io** | Fastest path from "I have a Dockerfile" to "it's on the internet" | Point them at your GitHub repo; they build the Dockerfile and give you a URL + HTTPS automatically. Best starting point. |
| **AWS App Runner / GCP Cloud Run / Azure Container Apps** | Same idea, if you're already in that cloud ecosystem | Pay-per-use, scales to zero when idle. |
| **AWS ECS/Fargate, GKE, AKS** | Larger teams, multiple services, more control | More setup, more to learn — don't start here. |
| A VPS you manage (DigitalOcean droplet, etc.) | Learning how servers actually work | You handle OS updates, HTTPS certs, restarts yourself. Educational, but more ongoing work. |

Any of the first two rows work well with the Dockerfile as-is.

### 2.2 HTTPS
Never serve real traffic over plain HTTP. All the platforms above give
you HTTPS automatically once you point a domain at them. If you're
running the container yourself on a VPS, put it behind a reverse proxy
(Caddy or nginx) that handles TLS termination — don't try to do TLS
inside uvicorn directly.

### 2.3 Authentication
**This project currently has none.** `/api/predict` is open to anyone
who can reach it. Before exposing this outside a trusted internal
network, add one of:
- **API key header** — simplest. Check a shared secret in a header
  (e.g. `X-API-Key`) via a FastAPI dependency, reject with 401 if
  missing/wrong.
- **OAuth2 / JWT** — if multiple client apps or user identities are
  involved. FastAPI has first-class support for this
  (`fastapi.security`).

### 2.4 Secrets management
`.env.example` shows the pattern: real secrets live in a `.env` file
(gitignored) locally, and in your hosting platform's "environment
variables" / "secrets" panel in production — **never committed to
git, never hardcoded in source.**

### 2.5 Retraining cadence
Per the PRD, the model's segment definitions are based on customer
behavior that will drift over time. A reasonable starting point:
- Re-run `train/train_pipeline.py` on fresh data quarterly (or
  whenever you have a meaningfully larger/newer dataset).
- Treat the output `models/*.joblib` files as build artifacts, not
  hand-edited files — regenerate them, don't patch them.
- Track accuracy over time (the script prints it) so a real drop is
  visible before it becomes a problem.

### 2.6 Monitoring, once you have real traffic
Start simple:
- Your hosting platform's built-in request/error/latency dashboard is
  usually enough at first (Render, Railway, Cloud Run all have one).
- If you outgrow that: ship logs to a log aggregator (e.g. Better
  Stack, Datadog, or your cloud's native logging), and consider
  tracking prediction-confidence distributions over time — a sudden
  shift can be an early signal of data drift, before accuracy
  visibly drops.

## 3. Continuous deployment (the step after CI)

`.github/workflows/ci.yml` currently lints, trains, tests, and builds
the Docker image on every push — but doesn't deploy anywhere yet, on
purpose (deployment targets are a per-person choice). To wire up real
continuous deployment, the pattern is:

1. On merge to `main`, after CI passes, push the built image to a
   registry (Docker Hub, GitHub Container Registry, or your cloud's
   registry).
2. Trigger your hosting platform to pull and run the new image. Render/
   Railway/Cloud Run can watch a registry or your repo directly and do
   this automatically — that's usually the easiest starting setup and
   needs no extra CI code.

## 4. Rollback

Keep at least the previous 2-3 Docker image versions available (most
registries do this by default via tags). If a deploy misbehaves,
redeploying the previous tag should take under a minute on any of the
platforms above — confirm this works *before* you need it in a real
incident.

## 5. A pre-launch checklist

- [ ] `python train/train_pipeline.py` run on real (not sample) data
- [ ] `pytest tests/ -v` passes
- [ ] `ruff check app/ common/ train/ tests/` passes
- [ ] `docker build .` succeeds locally
- [ ] Authentication added (Section 2.3) if this will be reachable from
      outside a trusted network
- [ ] HTTPS confirmed working
- [ ] Real secrets moved out of any `.env` file and into the platform's
      secret manager
- [ ] Health check endpoint confirmed reachable by your platform's
      monitor
- [ ] A plan for how/when the model gets retrained
