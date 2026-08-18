"""
tests/test_api.py
===================
Run with:  pytest -v

`TestClient` (from FastAPI/Starlette) lets us call our own API in-process
-- no real network socket, no separate `uvicorn` process needed. It
triggers the same startup event (`@app.on_event("startup")`) that a real
server would, so the model gets loaded exactly like it does in production.

This is the standard way to test a FastAPI service. In a real project,
these tests run automatically on every push (see .github/workflows/ci.yml).
"""

from fastapi.testclient import TestClient

from app.main import app

VALID_CUSTOMER = {
    "Year_Birth": 1978,
    "Education": "Graduation",
    "Marital_Status": "Married",
    "Income": 58000,
    "Kidhome": 0,
    "Teenhome": 1,
    "Dt_Customer": "15-03-2013",
    "Recency": 25,
    "MntWines": 450,
    "MntFruits": 30,
    "MntMeatProducts": 220,
    "MntFishProducts": 45,
    "MntSweetProducts": 20,
    "MntGoldProds": 60,
    "NumDealsPurchases": 3,
    "NumWebPurchases": 6,
    "NumCatalogPurchases": 4,
    "NumStorePurchases": 8,
    "NumWebVisitsMonth": 5,
    "AcceptedCmp1": 0,
    "AcceptedCmp2": 0,
    "AcceptedCmp3": 0,
    "AcceptedCmp4": 1,
    "AcceptedCmp5": 0,
}


def test_health_check():
    # `with TestClient(app) as client:` runs the startup event before the
    # first request and the shutdown event after the block exits.
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_homepage_serves_html():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


def test_predict_returns_valid_segment():
    with TestClient(app) as client:
        response = client.post("/api/predict", json=VALID_CUSTOMER)
        assert response.status_code == 200

        body = response.json()
        assert body["cluster"] in (0, 1, 2)
        assert body["segment_name"] in ("Low-Value", "Mid-Value", "High-Value")
        assert 0.0 <= body["confidence"] <= 1.0
        # Probabilities should sum to ~1
        assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_rejects_missing_fields():
    with TestClient(app) as client:
        response = client.post("/api/predict", json={"Year_Birth": 1978})
        assert response.status_code == 422  # FastAPI's automatic validation error


def test_predict_rejects_bad_types():
    with TestClient(app) as client:
        bad_customer = dict(VALID_CUSTOMER)
        bad_customer["Income"] = "not-a-number"
        response = client.post("/api/predict", json=bad_customer)
        assert response.status_code == 422


def test_predict_rejects_out_of_range_year():
    with TestClient(app) as client:
        bad_customer = dict(VALID_CUSTOMER)
        bad_customer["Year_Birth"] = 1500  # violates gt=1900 constraint
        response = client.post("/api/predict", json=bad_customer)
        assert response.status_code == 422


def test_low_spend_customer_predicts_low_value_segment():
    """A sanity/regression test: a clearly low-spending, low-income
    customer should land in the Low-Value segment. If a future change
    to the feature engineering or model breaks this basic pattern,
    this test should fail loudly."""
    low_value_customer = dict(VALID_CUSTOMER)
    low_value_customer.update({
        "Income": 18000,
        "MntWines": 2, "MntFruits": 1, "MntMeatProducts": 5,
        "MntFishProducts": 1, "MntSweetProducts": 1, "MntGoldProds": 2,
    })
    with TestClient(app) as client:
        response = client.post("/api/predict", json=low_value_customer)
        assert response.status_code == 200
        assert response.json()["segment_name"] == "Low-Value"
