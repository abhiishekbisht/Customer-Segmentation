"""
app/model_service.py
======================
Loading a ~few-MB model file from disk takes real time (tens to
hundreds of milliseconds). If we did that inside every request handler,
every single prediction would pay that cost.

Instead, this module loads everything ONCE, when the app process starts,
into module-level variables. Every request then just reuses the already-
loaded objects in memory. This is a standard pattern for any ML-serving
API, in FastAPI or otherwise.
"""

import json
from pathlib import Path

import pickle
import numpy as np
import pandas as pd

from app.schemas import CustomerInput
from common.feature_engineering import (
    FeatureEngineer,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelService:
    def __init__(self) -> None:
        self.feature_engineer: FeatureEngineer | None = None
        self.model = None
        self.segment_profiles: dict = {}

    def load(self) -> None:
        """Called once at startup (see app/main.py's startup event)."""
        fe_path = MODELS_DIR / "feature_engineer.pkl"
        model_path = MODELS_DIR / "catboost_model.pkl"
        profiles_path = MODELS_DIR / "segment_profiles.json"

        if not fe_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found in models/. "
                "Run `python train/train_pipeline.py` first."
            )

        with open(fe_path, "rb") as f:
            self.feature_engineer = pickle.load(f)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(profiles_path) as f:
            self.segment_profiles = json.load(f)

    def is_ready(self) -> bool:
        return self.model is not None and self.feature_engineer is not None

    def predict(self, customer: CustomerInput) -> dict:
        """
        Turns one validated CustomerInput into a prediction dict.
        Raises RuntimeError if called before load().
        """
        if not self.is_ready():
            raise RuntimeError("ModelService.load() must be called before predict().")

        # Pydantic model -> single-row DataFrame, using the exact same,
        # column names the raw CSV had (that's what FeatureEngineer expects).
        raw_df = pd.DataFrame([customer.model_dump()])

        # SAME feature engineering used in training. No drift possible,
        # because it's literally the same fitted object, loaded from disk.
        X = self.feature_engineer.transform(raw_df)

        # CatBoost's multiclass .predict() returns a 2D array (e.g. shape
        # (1, 1)) even for a single row, so we flatten before extracting
        # the scalar -- calling int() directly on a non-0-d array raises
        # "only 0-dimensional arrays can be converted to Python scalars".
        cluster = int(np.asarray(self.model.predict(X)).ravel()[0])
        proba = self.model.predict_proba(X)[0]
        class_labels = self.model.classes_

        probabilities = {
            str(int(c)): round(float(p), 4)
            for c, p in zip(class_labels, proba, strict=True)
        }
        confidence = probabilities[str(cluster)]

        profile = self.segment_profiles.get(str(cluster), {})
        segment_name = profile.get("segment_name", f"Segment {cluster}")

        return {
            "cluster": cluster,
            "segment_name": segment_name,
            "confidence": confidence,
            "probabilities": probabilities,
        }


# One shared instance, imported by app/main.py. This module-level
# singleton pattern is how you avoid re-loading the model per request.
model_service = ModelService()
