"""
app/schemas.py
===============
FastAPI's headline feature (the thing that makes it different from
Flask) is that it uses Python type hints + Pydantic to describe exactly
what a request body should look like, and it will:
  1. Validate incoming JSON against this shape automatically.
  2. Reject bad requests with a clear 422 error BEFORE your code runs.
  3. Auto-generate interactive docs (/docs) from these classes.

You never write "if not request.json.get('income'): return error" by
hand -- you just describe the data once, here, as a class.

`BaseModel` is Pydantic's base class. Every attribute becomes a field.
`Field(...)` lets us attach validation rules (min/max, examples, docs).
"""

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """
    One customer record, in the SAME raw shape as a row of
    marketing_campaign.csv. This is what a client (our web form, or
    another service) sends us. FastAPI will automatically reject a
    request that's missing a field or has the wrong type.
    """

    # `...` (Ellipsis) means "required, no default value".
    # `Field(..., gt=1900, le=2015, description=...)` adds validation +
    # documentation that shows up in the auto-generated /docs page.
    Year_Birth: int = Field(..., gt=1900, le=2015, description="Customer's birth year")
    Education: str = Field(..., description="One of: Basic, 2n Cycle, Graduation, Master, PhD")
    Marital_Status: str = Field(..., description="e.g. Single, Married, Together, Divorced, Widow")
    Income: float = Field(..., ge=0, description="Yearly household income")
    Kidhome: int = Field(..., ge=0, le=10, description="Number of young children at home")
    Teenhome: int = Field(..., ge=0, le=10, description="Number of teenagers at home")
    Dt_Customer: str = Field(..., description="Enrollment date, format DD-MM-YYYY")
    Recency: int = Field(..., ge=0, description="Days since last purchase")

    MntWines: float = Field(..., ge=0)
    MntFruits: float = Field(..., ge=0)
    MntMeatProducts: float = Field(..., ge=0)
    MntFishProducts: float = Field(..., ge=0)
    MntSweetProducts: float = Field(..., ge=0)
    MntGoldProds: float = Field(..., ge=0)

    NumDealsPurchases: int = Field(..., ge=0)
    NumWebPurchases: int = Field(..., ge=0)
    NumCatalogPurchases: int = Field(..., ge=0)
    NumStorePurchases: int = Field(..., ge=0)
    NumWebVisitsMonth: int = Field(..., ge=0)

    AcceptedCmp1: int = Field(0, ge=0, le=1)
    AcceptedCmp2: int = Field(0, ge=0, le=1)
    AcceptedCmp3: int = Field(0, ge=0, le=1)
    AcceptedCmp4: int = Field(0, ge=0, le=1)
    AcceptedCmp5: int = Field(0, ge=0, le=1)

    # `model_config` with `json_schema_extra` gives FastAPI's /docs page
    # a "Try it out" example filled in automatically -- purely for
    # developer experience, not required for validation to work.
    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionOutput(BaseModel):
    """What we send back. Declaring this (and setting it as a route's
    `response_model`) means FastAPI will also validate and document
    OUR response shape, not just the request."""

    cluster: int
    segment_name: str
    confidence: float
    probabilities: dict[str, float]
