"""
train/train_pipeline.py
========================
This script is the "offline" half of the project. Run it once (and
again whenever you get fresh data) to produce the model files the API
serves. It is a consolidated, bug-fixed version of your two notebooks:

  1. Feature_engineering_and_clustering.ipynb  -> builds features,
     clusters customers into segments (KMeans on PCA-reduced data).
  2. Feature_engineering_and_Classification.ipynb -> trains a CatBoost
     classifier to predict which segment a customer belongs to, so
     that in production we DON'T need to re-run clustering (which needs
     the whole customer base) for every single prediction -- we just
     need the fast, already-trained classifier.

Run it with:
    python train/train_pipeline.py

It writes three files into models/:
    feature_engineer.joblib   - the fitted FeatureEngineer (common/feature_engineering.py)
    catboost_model.joblib     - the trained, tuned CatBoostClassifier
    segment_profiles.json     - human-readable name + averages for each cluster,
                                 used by the API to turn "cluster 2" into
                                 something a user can read.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Make `common/` importable when this script is run directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.feature_engineering import (
    FEATURE_ORDER,
    OUTLIER_COLUMNS,
    FeatureEngineer,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "notebooks" / "marketing_campaign.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def load_raw_data() -> pd.DataFrame:
    print(f"Loading raw data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, sep="\t")
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def engineer_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureEngineer]:
    print("Fitting FeatureEngineer and building the 21-feature table ...")
    fe = FeatureEngineer().fit(raw_df)
    X = fe.transform(raw_df)
    print(f"  Engineered feature table shape: {X.shape}")
    return X, fe


def run_clustering(X: pd.DataFrame) -> np.ndarray:
    """
    Reproduces the clustering notebook: scale numeric features, power-
    transform the skewed/outlier ones, reduce to 2 components with PCA,
    then KMeans(k=3) -- which the notebook picked via silhouette score.

    This ONLY runs offline, on the full historical dataset, to invent the
    "ground truth" segment labels that the classifier then learns to
    predict quickly for one customer at a time.
    """
    print("Running clustering to generate segment labels (KMeans on PCA) ...")
    numeric_features = [c for c in FEATURE_ORDER if c not in OUTLIER_COLUMNS]

    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("scale", StandardScaler()),
            ]), numeric_features),
            ("outlier_prone", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("power", PowerTransformer(standardize=True)),
            ]), OUTLIER_COLUMNS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    scaled = preprocessor.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(reduced)

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"  Cluster sizes: {dict(zip(unique_labels.tolist(), counts.tolist(), strict=True))}")
    return labels


def train_classifier(X: pd.DataFrame, y: np.ndarray) -> CatBoostClassifier:
    print("Training CatBoostClassifier to predict segment from raw features ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    base_model = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    param_dist = {
        "iterations": [200, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Best params: {search.best_params_}")
    print(f"  Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Refit on ALL data for the final artifact that ships to production.
    best_model.fit(X, y)
    return best_model


def build_segment_profiles(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Human-readable summary per cluster, so the API can return more
    than just a bare integer to whoever calls it."""
    profile_df = X.copy()
    profile_df["cluster"] = y

    profiles = {}
    for cluster_id, group in profile_df.groupby("cluster"):
        profiles[str(int(cluster_id))] = {
            "size": len(group),
            "avg_income": round(float(group["Income"].mean()), 2),
            "avg_total_spending": round(float(group["Total_Spending"].mean()), 2),
            "avg_age": round(float(group["Age"].mean()), 1),
        }

    # Rank clusters by spending to assign readable labels.
    ranked = sorted(profiles.items(), key=lambda kv: kv[1]["avg_total_spending"])
    tier_names = ["Low-Value", "Mid-Value", "High-Value"]
    for tier_name, (_cluster_id, info) in zip(tier_names, ranked, strict=True):
        info["segment_name"] = tier_name

    return profiles


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    raw_df = load_raw_data()
    X, feature_engineer = engineer_features(raw_df)
    y = run_clustering(X)
    model = train_classifier(X, y)
    profiles = build_segment_profiles(X, y)

    import pickle

    with open(MODELS_DIR / "feature_engineer.pkl", "wb") as f:
        pickle.dump(feature_engineer, f)
    with open(MODELS_DIR / "catboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "segment_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    print("\nSaved artifacts to models/:")
    print("  - feature_engineer.pkl")
    print("  - catboost_model.pkl")
    print("  - segment_profiles.json")
    print("\nDone. You can now start the API with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
