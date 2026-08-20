"""main.py - MLB Pitch Classifier FastAPI 백엔드"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from explain import generate_explanation
from model_utils import predict

app = FastAPI(title="MLB Pitch Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PitchFeatures(BaseModel):
    release_speed: float
    release_spin_rate: float
    release_extension: float
    release_pos_x: float
    release_pos_z: float
    pfx_x: float
    pfx_z: float
    plate_x: float
    plate_z: float
    vx0: float
    vy0: float
    vz0: float
    ax: float
    ay: float
    az: float
    effective_speed: float
    spin_axis: float


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    explanation: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_pitch(features: PitchFeatures):
    input_dict = features.model_dump()
    label, confidence, proba = predict(input_dict)
    explanation = generate_explanation(input_dict, label, confidence, proba)
    return PredictionResponse(
        predicted_label=label,
        confidence=confidence,
        probabilities=proba,
        explanation=explanation,
    )
