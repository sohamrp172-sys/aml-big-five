"""FastAPI inference endpoint for Big Five personality prediction."""

import sys
from pathlib import Path

# Ensure project root is on sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.data_loader import load_data, ITEM_COLS, REVERSE_SCORED
from src.factor_analysis import fit_factor_analysis, get_factor_scores

app = FastAPI(title="Big Five Personality Analyzer")

# Factor-to-label mapping (order matches varimax rotation output)
FACTOR_LABELS = {
    "Factor1": "Extraversion",
    "Factor2": "Neuroticism",
    "Factor3": "Agreeableness",
    "Factor4": "Conscientiousness",
    "Factor5": "Openness",
}

fa_model = None


@app.on_event("startup")
def startup():
    global fa_model
    print("Loading data and fitting Factor Analysis model...")
    df = load_data()
    fa_model = fit_factor_analysis(df, n_factors=5)
    print("Model ready.")


class PredictRequest(BaseModel):
    EXT1: int; EXT2: int; EXT3: int; EXT4: int; EXT5: int
    EXT6: int; EXT7: int; EXT8: int; EXT9: int; EXT10: int
    EST1: int; EST2: int; EST3: int; EST4: int; EST5: int
    EST6: int; EST7: int; EST8: int; EST9: int; EST10: int
    AGR1: int; AGR2: int; AGR3: int; AGR4: int; AGR5: int
    AGR6: int; AGR7: int; AGR8: int; AGR9: int; AGR10: int
    CSN1: int; CSN2: int; CSN3: int; CSN4: int; CSN5: int
    CSN6: int; CSN7: int; CSN8: int; CSN9: int; CSN10: int
    OPN1: int; OPN2: int; OPN3: int; OPN4: int; OPN5: int
    OPN6: int; OPN7: int; OPN8: int; OPN9: int; OPN10: int

    @field_validator("*")
    @classmethod
    def check_range(cls, v):
        if v < 1 or v > 5:
            raise ValueError("Each response must be between 1 and 5")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if fa_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    data = req.model_dump()

    # Apply reverse scoring (same as data_loader)
    for col in REVERSE_SCORED:
        data[col] = 6 - data[col]

    # Build single-row DataFrame in correct column order
    df_input = pd.DataFrame([data])[ITEM_COLS]

    scores = get_factor_scores(fa_model, df_input)

    result = {
        FACTOR_LABELS[col]: round(float(scores[col].iloc[0]), 4)
        for col in scores.columns
    }
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
