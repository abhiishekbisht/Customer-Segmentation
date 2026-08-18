"""
common/feature_engineering.py
==============================
This module turns a RAW customer record (the same shape as a row in
marketing_campaign.csv) into the 21 engineered features the model was
trained on.

WHY THIS FILE IS SEPARATE (important production concept):
------------------------------------------------------------
The #1 cause of "it worked in the notebook but broke in production" bugs
is that training code and serving code silently drift apart -- someone
tweaks the notebook, forgets to update the API, and now the model gets
fed data it's never seen.

We solve that by writing the feature engineering EXACTLY ONCE, in this
file, and importing it from both:
  - train/train_pipeline.py   (fits it on historical data, saves it)
  - app/model_service.py      (loads it, applies it to live requests)

There is only ever one source of truth for "how do we turn raw fields
into model features".
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Fixed lookup tables. These encode business rules discovered during EDA
# in `EDA.ipynb` / `Feature_engineering_and_clustering.ipynb`.
# ----------------------------------------------------------------------

# Ordinal mapping: higher number = more formal education.
EDUCATION_MAP: dict[str, int] = {
    "Basic": 0,
    "2n Cycle": 1,
    "Graduation": 2,
    "Master": 3,
    "PhD": 4,
}
 
# Binary mapping: does the customer live with a partner?
MARITAL_MAP: dict[str, int] = {
    "Married": 1,
    "Together": 1,
    "Single": 0,
    "Divorced": 0,
    "Widow": 0,
    "Alone": 0,
    "Absurd": 0,
    "YOLO": 0,
}

# Final feature order the CatBoost model expects. Order matters for some
# model formats, and keeping an explicit list makes bugs loud (a missing
# column raises a KeyError instead of silently shifting every value over).
FEATURE_ORDER: list[str] = [
    "Age",
    "Education",
    "Marital Status",
    "Parental Status",
    "Children",
    "Income",
    "Total_Spending",
    "Days_as_Customer",
    "Recency",
    "Wines",
    "Fruits",
    "Meat",
    "Fish",
    "Sweets",
    "Gold",
    "Web",
    "Catalog",
    "Store",
    "Discount Purchases",
    "Total Promo",
    "NumWebVisitsMonth",
]

# Columns that get IQR outlier-capping applied (per the EDA notebook).
OUTLIER_COLUMNS: list[str] = [
    "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Age", "Total_Spending",
]


class FeatureEngineer:
    """
    Reproduces the cleaning + feature engineering from
    `Feature_engineering_and_clustering.ipynb`, with two production fixes:

    1. FIXED BUG: the notebook did
           df["Education"] = df["Education"].replace({...}, inplace=True)
       `inplace=True` makes `.replace()` return `None`, so that line wipes
       the column to all-None. Below we do the replace WITHOUT reassigning
       (or without inplace=True) so the mapping actually sticks.

    2. FIXED LEAKAGE RISK: outlier caps (IQR bounds) and the Income median
       are computed ONCE on the training set inside `fit()` and reused
       for every future prediction, instead of being recomputed on
       whatever data happens to be present (which is what the notebook
       function did, and which would make one customer's prediction
       depend on which other customers happened to be in the batch).

    Usage:
        fe = FeatureEngineer().fit(raw_training_df)
        X = fe.transform(raw_new_df)
    """

    def __init__(self) -> None:
        self.income_median_: float | None = None
        self.iqr_bounds_: dict[str, tuple] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # fit: learn statistics from TRAINING data only
    # ------------------------------------------------------------------
    def fit(self, raw_df: pd.DataFrame) -> FeatureEngineer:
        df = raw_df.copy()

        # Learn the Income median from training data (used to fill
        # missing Income values at prediction time too).
        self.income_median_ = float(df["Income"].median())

        # Build the engineered columns once so we can compute IQR bounds
        # on the *engineered* Age / Total_Spending values.
        engineered = self._engineer(df, income_median=self.income_median_)

        # Learn IQR bounds for every outlier-prone column.
        for col in OUTLIER_COLUMNS:
            q1 = engineered[col].quantile(0.25)
            q3 = engineered[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.iqr_bounds_[col] = (lower, upper)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform: apply the SAME logic to any new data (training or live)
    # ------------------------------------------------------------------
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer.fit() must be called before transform().")

        engineered = self._engineer(raw_df.copy(), income_median=self.income_median_)

        # Apply the frozen outlier caps learned during fit().
        for col, (lower, upper) in self.iqr_bounds_.items():
            engineered[col] = engineered[col].clip(lower=lower, upper=upper)

        # Guarantee column order matches what the model was trained on.
        return engineered[FEATURE_ORDER]

    def fit_transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(raw_df).transform(raw_df)

    # ------------------------------------------------------------------
    # Internal: the actual feature engineering logic (mirrors the notebook)
    # ------------------------------------------------------------------
    @staticmethod
    def _engineer(df: pd.DataFrame, income_median: float) -> pd.DataFrame:
        df = df.copy()

        # --- Clean -----------------------------------------------------
        df["Income"] = df["Income"].fillna(income_median)

        for col in ["ID", "Z_CostContact", "Z_Revenue"]:
            if col in df.columns:
                df = df.drop(columns=col)

        # --- Derived fields ---------------------------------------------
        current_year = datetime.today().year
        df["Age"] = current_year - df["Year_Birth"]

        # FIX: no inplace=True, and we assign the RETURN VALUE of .map()
        # This is the corrected version of the notebook's buggy line.
        df["Education"] = df["Education"].map(EDUCATION_MAP)
        df["Marital Status"] = df["Marital_Status"].map(MARITAL_MAP)

        df["Children"] = df["Kidhome"] + df["Teenhome"]
        df["Parental Status"] = np.where(df["Children"] > 0, 1, 0)

        df["Total_Spending"] = (
            df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
            + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
        )

        df["Total Promo"] = (
            df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"]
            + df["AcceptedCmp4"] + df["AcceptedCmp5"]
        )

        dt_customer = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
        # Fall back to a generic parser for any format the strict parser missed.
        mask = dt_customer.isna()
        if mask.any():
            dt_customer.loc[mask] = pd.to_datetime(df.loc[mask, "Dt_Customer"], errors="coerce")
        df["Days_as_Customer"] = (datetime.today() - dt_customer).dt.days

        # --- Rename to match training column names ----------------------
        df = df.rename(columns={
            "MntWines": "Wines",
            "MntFruits": "Fruits",
            "MntMeatProducts": "Meat",
            "MntFishProducts": "Fish",
            "MntSweetProducts": "Sweets",
            "MntGoldProds": "Gold",
            "NumWebPurchases": "Web",
            "NumCatalogPurchases": "Catalog",
            "NumStorePurchases": "Store",
            "NumDealsPurchases": "Discount Purchases",
        })

        return df[FEATURE_ORDER]
