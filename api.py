import pathlib

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List


# =========================================================
# PATHS
# =========================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# =========================================================
# FEATURE COLUMNS
# =========================================================
#
# Must match FEATURE_COLS in election_model.py and
# visualization/election_charts.py EXACTLY - same columns,
# same order.
#
# =========================================================

FEATURE_COLS = [
    "state_encoded",
    "demographic_encoded",
    "turnout_2016",
    "turnout_2021",
    "turnout_change_21",
    "margin_2016",
    "swing_2016",
    "result_2016_encoded",
]


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Election Prediction API",
    description=(
        "Serves predictions from the trained margin, "
        "retention and party models. Streamlit no longer "
        "loads the .pkl files directly - it calls this "
        "service instead."
    ),
)


_models = {}


# =========================================================
# LOAD MODELS ON STARTUP
# =========================================================

@app.on_event("startup")
def load_models():

    required_models = [
        "rf_margin_model.pkl",
        "rf_retained_model.pkl",
        "rf_party_model.pkl",
    ]

    missing_models = [
        filename
        for filename in required_models
        if not (MODEL_DIR / filename).exists()
    ]

    if missing_models:

        raise RuntimeError(
            "Missing model file(s): "
            + ", ".join(missing_models)
            + f"\nExpected location: {MODEL_DIR}"
        )

    _models["margin"] = joblib.load(
        MODEL_DIR / "rf_margin_model.pkl"
    )

    _models["retained"] = joblib.load(
        MODEL_DIR / "rf_retained_model.pkl"
    )

    _models["party"] = joblib.load(
        MODEL_DIR / "rf_party_model.pkl"
    )

    print("Models loaded from:", MODEL_DIR)


# =========================================================
# REQUEST / RESPONSE SCHEMAS
# =========================================================

class PredictionRequest(BaseModel):

    rows: List[List[float]] = Field(
        ...,
        description=(
            "Feature rows, each in FEATURE_COLS order: "
            + ", ".join(FEATURE_COLS)
        ),
    )


class PredictionResponse(BaseModel):

    margin: List[float]
    retained: List[int]
    party: List[str]


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "models_loaded": len(_models) == 3,
    }


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    if len(_models) != 3:

        raise HTTPException(
            status_code=503,
            detail="Models are not loaded yet.",
        )

    if not request.rows:

        raise HTTPException(
            status_code=400,
            detail="No feature rows were provided.",
        )

    for row in request.rows:

        if len(row) != len(FEATURE_COLS):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Each row must have "
                    f"{len(FEATURE_COLS)} values matching: "
                    + ", ".join(FEATURE_COLS)
                ),
            )

    features_df = pd.DataFrame(
        request.rows,
        columns=FEATURE_COLS,
    )

    margin_predictions = (
        _models["margin"]
        .predict(features_df)
        .tolist()
    )

    retained_predictions = (
        _models["retained"]
        .predict(features_df)
        .tolist()
    )

    party_predictions = (
        _models["party"]
        .predict(features_df)
        .tolist()
    )

    return PredictionResponse(
        margin=margin_predictions,
        retained=retained_predictions,
        party=party_predictions,
    )


# =========================================================
# ENTRYPOINT (optional - lets you run `python api/main.py`)
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )