from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from risk_aware.domain.schemas import ComplaintRequest, InferenceOutput
from risk_aware.models.thresholds import ThresholdPolicy
from risk_aware.pipelines.inference import InferenceOrchestrator
from risk_aware.utils.serialization import load_artifact

app = FastAPI(title="RiskAware Complaints Engine", version="0.1.0")


def load_orchestrator(model_dir: str = "artifacts/models", threshold_dir: str = "artifacts/thresholds") -> InferenceOrchestrator:
    category_model = load_artifact(Path(model_dir) / "category_model.joblib")
    legal_model = load_artifact(Path(model_dir) / "legal_model.joblib")
    priority_model = load_artifact(Path(model_dir) / "priority_model.joblib")
    threshold_payload = load_artifact(Path(threshold_dir) / "priority_threshold.joblib")
    threshold_policy = ThresholdPolicy(
        mode=str(threshold_payload["mode"]),
        threshold_p1=float(threshold_payload["threshold_p1"]),
    )
    return InferenceOrchestrator(
        category_model=category_model,
        legal_model=legal_model,
        priority_model=priority_model,
        threshold_policy=threshold_policy,
    )


_orchestrator: InferenceOrchestrator | None = None


@app.on_event("startup")
def startup_event() -> None:
    global _orchestrator
    try:
        _orchestrator = load_orchestrator()
    except FileNotFoundError:
        _orchestrator = None


@app.get("/health")
def health() -> dict[str, str]:
    status = "ready" if _orchestrator is not None else "missing_artifacts"
    return {"status": status}


@app.post("/predict", response_model=InferenceOutput)
def predict(payload: ComplaintRequest) -> InferenceOutput:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not available.")
    return _orchestrator.predict(payload)

