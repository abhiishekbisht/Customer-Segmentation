# Customer Segmentation & Value Prediction API

An end-to-end Machine Learning web application and RESTful API that predicts a customer's value segment (**Low**, **Mid**, or **High**) based on demographic profiles, purchasing behaviors, and engagement channels.

Built with **FastAPI**, **CatBoost**, **Scikit-learn**, and **Docker**.

---

## 🌟 Key Features

- **Predictive ML Model:** Powered by a high-accuracy **CatBoost Classifier** trained to categorize customers into value segments.
- **Interactive Web Interface:** User-friendly browser form for real-time predictions (`GET /`).
- **RESTful API Endpoint:** Standardized JSON endpoint (`POST /predict`) for programmatically scoring customer data.
- **Zero Training-Serving Skew:** Centralized feature engineering pipeline shared identically between offline training and online serving.
- **Interactive API Documentation:** Built-in Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Automated Testing & CI:** Full test suite powered by `pytest` with automated GitHub Actions CI integration.
- **Production-Ready Containerization:** Dockerized microservice setup with `docker-compose`.

---

## 📂 Project Structure

```
.
├── app/                        # FastAPI Web Application & Service Layer
│   ├── main.py                 # Application entry point & API route handlers
│   ├── model_service.py        # ML model wrapper and inference engine
│   ├── schemas.py              # Pydantic data schemas for request validation
│   ├── static/
│   │   └── style.css           # Styling for the web UI console
│   └── templates/
│       └── index.html          # Interactive Web UI console template
├── common/                     # Core Business Logic & Shared Utilities
│   └── feature_engineering.py  # Single source of truth for feature transformations
├── models/                     # Trained ML model binaries & segment metadata
│   └── segment_profiles.json   # Value segment profiles description
├── tests/                      # Automated Unit & Integration Tests
│   └── test_api.py             # FastAPI endpoint and inference test cases
├── train/                      # Training Pipeline & Model Generation
│   └── train_pipeline.py       # Standalone training script to train & export models
├── notebooks/                  # Jupyter Notebooks for EDA & ML Modeling
│   ├── EDA.ipynb
│   ├── Feature_engineering_and_clustering.ipynb
│   └── Feature_engineering_and_Classification.ipynb
├── DEPLOYMENT.md               # Guide for deploying to cloud environments
├── Dockerfile                  # Container build instructions
├── STEP_BY_STEP_GUIDE.md       # Guided code walkthrough and architecture breakdown
├── docker-compose.yml          # Container orchestration configuration
├── pyproject.toml              # Project metadata & tool configurations
├── requirements.txt            # Production Python dependencies
└── requirements-dev.txt        # Development & testing dependencies
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Virtual Environment

Ensure you have **Python 3.10+** installed.

```bash
# Clone the repository
git clone https://github.com/abhiishekbisht/Customer-Segmentation.git
cd Customer-Segmentation

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 2. Train the Model

Before running the API for the first time, generate the model artifacts by executing the training pipeline:

```bash
python train/train_pipeline.py
```

*This reads the dataset, runs feature engineering, trains the CatBoost model, and exports binaries to the `models/` directory.*

### 3. Launch the API Server

Start the FastAPI application with live-reloading:

```bash
uvicorn app.main:app --reload
```

### 4. Access the Endpoints

- **Web Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🧪 Testing

Run the automated test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

You can run the application seamlessly using Docker and Docker Compose:

```bash
# Ensure models exist by running the training pipeline locally first
python train/train_pipeline.py

# Build and start the container
docker compose up --build
```

Access the containerized web UI at [http://localhost:8000](http://localhost:8000).

---

## ☁️ Cloud Deployment (Free PaaS)

### Option 1: Deploy to Render (Recommended - 1 Click Blueprint)

1. Log in to [Render.com](https://render.com/).
2. Click **New +** -> **Blueprints**.
3. Connect your GitHub repository: `https://github.com/abhiishekbisht/Customer-Segmentation`.
4. Render will automatically detect [`render.yaml`](file:///Users/datalynx/Downloads/customer-segmentation-api/render.yaml), build your Docker container, and deploy your live URL with free HTTPS.

### Option 2: Deploy to Koyeb (Free Docker Cloud Hosting)

1. Log in to [Koyeb.com](https://app.koyeb.com/).
2. Click **Create Service** -> Select **GitHub**.
3. Choose your repository: `abhiishekbisht/Customer-Segmentation`.
4. Select **Builder: Dockerfile** (it will auto-detect your `Dockerfile` on port `8000`).
5. Click **Deploy**! Koyeb will build and host your API with a free `.koyeb.app` HTTPS domain.

### Option 3: Deploy to Railway

1. Log in to [Railway.app](https://railway.app/).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select `abhiishekbisht/Customer-Segmentation`.
4. Railway will build the `Dockerfile` automatically and provide a public URL.

---

## 📚 Documentation

For additional guides:
- **Architecture & Guided Walkthrough:** See [`STEP_BY_STEP_GUIDE.md`](file:///Users/datalynx/Downloads/customer-segmentation-api/STEP_BY_STEP_GUIDE.md)
- **Cloud & Production Deployment:** See [`DEPLOYMENT.md`](file:///Users/datalynx/Downloads/customer-segmentation-api/DEPLOYMENT.md)
