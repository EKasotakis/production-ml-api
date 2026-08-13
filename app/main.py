import joblib
import pandas as pd
import logging
import time

from fastapi import FastAPI

from app.schemas import CustomerInput, PredictionResponse


MODEL_PATH = "models/churn_model.joblib"

metrics = {
    "total_predictions": 0,
    "predicted_churn": 0,
    "predicted_stay": 0,
    "probability_sum": 0.0,
    "latency_sum_ms": 0.0
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    start_time = time.perf_counter()

    try:
        input_df = pd.DataFrame([customer.model_dump()])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0, 1]

        latency_ms = (time.perf_counter() - start_time) * 1000

        metrics["total_predictions"] += 1

        if prediction == 1:
            metrics["predicted_churn"] += 1
        else:
            metrics["predicted_stay"] += 1

        metrics["probability_sum"] += float(probability)
        metrics["latency_sum_ms"] += latency_ms

        logger.info(
            "prediction=%s probability=%.4f latency_ms=%.2f",
            int(prediction),
            probability,
            latency_ms
        )

        return PredictionResponse(
            churn_prediction=bool(prediction),
            churn_probability=float(probability)
        )

    except Exception:
        logger.exception("Prediction failed")
        raise


@app.get("/metrics")
def get_metrics():
    total = metrics["total_predictions"]

    if total == 0:
        return {
            "total_predictions": 0,
            "predicted_churn": 0,
            "predicted_stay": 0,
            "average_churn_probability": 0.0,
            "average_latency_ms": 0.0
        }

    return {
        "total_predictions": total,
        "predicted_churn": metrics["predicted_churn"],
        "predicted_stay": metrics["predicted_stay"],
        "average_churn_probability": metrics["probability_sum"] / total,
        "average_latency_ms": metrics["latency_sum_ms"] / total
    }