import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DATA_PATH = "data/smart_manufacturing_data.csv"
MODEL_PATH = "models/machineguard_anomaly_model.joblib"

FEATURES = [
    "machine_id", "temperature", "vibration", "humidity", "pressure",
    "energy_consumption", "hour", "day_of_week",
]

app = FastAPI(
    title="MachineGuard AI",
    description="Predictive machine condition monitoring",
    version="2.0.0",
)

if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"Dataset not found: {DATA_PATH}")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found: {MODEL_PATH}. Run: python backend\\train_model.py")

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
MODEL_THRESHOLD = model_package.get("threshold", 0.50)

# Each machine has its own current position in its historical rows.
machine_rows = {
    int(machine_id): group.sort_values("timestamp").reset_index(drop=True)
    for machine_id, group in df.groupby("machine_id")
}
machine_positions = {machine_id: 0 for machine_id in machine_rows}

analysis_history = []
active_alerts = []
last_updated = {}


class AnalysisRequest(BaseModel):
    machine_id: int


class SimulationRequest(BaseModel):
    machine_id: int
    temperature: Optional[float] = None
    vibration: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    energy_consumption: Optional[float] = None


def _clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value.item() if hasattr(value, "item") else value


def _row_dict(row):
    return {key: _clean_value(value) for key, value in row.to_dict().items()}


def get_current_row(machine_id: int):
    if machine_id not in machine_rows:
        raise HTTPException(status_code=404, detail="Machine not found")
    position = machine_positions[machine_id]
    return machine_rows[machine_id].iloc[position].copy()


def machine_payload(machine_id: int):
    row = get_current_row(machine_id)
    ai = risk_from_row(row)
    return {
        "machine_id": int(row["machine_id"]),
        "temperature": round(float(row["temperature"]), 2),
        "vibration": round(float(row["vibration"]), 2),
        "humidity": round(float(row["humidity"]), 2),
        "pressure": round(float(row["pressure"]), 2),
        "energy_consumption": round(float(row["energy_consumption"]), 2),
        "machine_status": int(row["machine_status"]),
        "anomaly_flag": int(row["anomaly_flag"]),
        "anomaly": int(row["anomaly_flag"]),
        "failure_type": str(row["failure_type"]),
        "maintenance_required": int(row["maintenance_required"]),
        "predicted_remaining_life": round(float(row["predicted_remaining_life"]), 2),
        "downtime_risk": float(row["downtime_risk"]),
        "timestamp": str(row["timestamp"]),
        "record_index": int(machine_positions[machine_id] + 1),
        "total_machine_records": int(len(machine_rows[machine_id])),
        "last_updated": last_updated.get(machine_id),
        "risk_score": ai["risk_score"],
        "condition": ai["condition"],
        "anomaly_probability": ai["anomaly_probability"],
    }


def advance_machine(machine_id: int):
    if machine_id not in machine_rows:
        raise HTTPException(status_code=404, detail="Machine not found")
    total = len(machine_rows[machine_id])
    current = machine_positions[machine_id]
    machine_positions[machine_id] = (current + 1) % total
    last_updated[machine_id] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    return machine_payload(machine_id)


def risk_from_row(row):
    timestamp = pd.to_datetime(row["timestamp"])
    input_data = pd.DataFrame([{
        "machine_id": int(row["machine_id"]),
        "temperature": float(row["temperature"]),
        "vibration": float(row["vibration"]),
        "humidity": float(row["humidity"]),
        "pressure": float(row["pressure"]),
        "energy_consumption": float(row["energy_consumption"]),
        "hour": int(timestamp.hour),
        "day_of_week": int(timestamp.dayofweek),
    }])
    probability = float(model.predict_proba(input_data)[0][1])
    prediction = int(probability >= MODEL_THRESHOLD)
    risk_score = round(probability * 100, 1)
    if risk_score >= 75:
        condition = "Critical"
    elif risk_score >= 50:
        condition = "Warning"
    elif risk_score >= 25:
        condition = "Watch"
    else:
        condition = "Healthy"

    importance = model.feature_importances_
    factor_data = []
    for feature, score in zip(FEATURES, importance):
        if feature in ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]:
            factor_data.append({
                "factor": feature,
                "value": round(float(row[feature]), 2),
                "importance": round(float(score), 4),
            })
    factor_data.sort(key=lambda x: x["importance"], reverse=True)
    top_factors = factor_data[:3]

    if condition == "Critical":
        recommendation = "Immediate maintenance inspection recommended. Check temperature and vibration conditions before continued operation."
    elif condition == "Warning":
        recommendation = "Schedule a maintenance inspection soon and closely monitor machine condition."
    elif condition == "Watch":
        recommendation = "Continue monitoring sensor trends for signs of deterioration."
    else:
        recommendation = "Machine condition appears normal. Continue routine monitoring."

    return {
        "machine_id": int(row["machine_id"]),
        "prediction": prediction,
        "anomaly_probability": round(probability, 4),
        "risk_score": risk_score,
        "condition": condition,
        "recommendation": recommendation,
        "timestamp": str(timestamp),
        "key_factors": top_factors,
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "dataset_rows": len(df),
        "machines": len(machine_rows),
        "model_loaded": True,
    }


@app.get("/api/dashboard")
def dashboard():
    current = pd.DataFrame([get_current_row(mid) for mid in sorted(machine_rows)])
    return {
        "total_machines": len(machine_rows),
        "total_records": len(df),
        "anomalies": int(df["anomaly_flag"].sum()),
        "maintenance_required": int(df["maintenance_required"].sum()),
        "machines_with_anomaly": int(current["anomaly_flag"].sum()),
        "latest_timestamp": str(df["timestamp"].max()),
    }


@app.get("/api/machines")
def machines():
    return [machine_payload(mid) for mid in sorted(machine_rows)]


@app.get("/api/machines/{machine_id}")
def machine_details(machine_id: int):
    return machine_payload(machine_id)


@app.get("/api/machines/{machine_id}/sensor-data")
def sensor_data(machine_id: int, limit: int = 100):
    if machine_id not in machine_rows:
        raise HTTPException(status_code=404, detail="Machine not found")
    machine = machine_rows[machine_id]
    start = max(0, machine_positions[machine_id] - limit + 1)
    rows = machine.iloc[start:machine_positions[machine_id] + 1]
    if len(rows) < limit:
        rows = machine.tail(limit)
    return [{
        "timestamp": str(row["timestamp"]),
        "temperature": round(float(row["temperature"]), 2),
        "vibration": round(float(row["vibration"]), 2),
        "humidity": round(float(row["humidity"]), 2),
        "pressure": round(float(row["pressure"]), 2),
        "energy_consumption": round(float(row["energy_consumption"]), 2),
    } for _, row in rows.iterrows()]


@app.post("/api/machines/{machine_id}/update")
def update_machine(machine_id: int):
    # One click advances only this machine to its next historical observation.
    return advance_machine(machine_id)


@app.post("/api/machines/update-all")
def update_all_machines():
    updated = [advance_machine(mid) for mid in sorted(machine_rows)]

    # Refresh the current alert view from the newly arrived records.
    refreshed_alerts = []
    for machine_id in sorted(machine_rows):
        result = risk_from_row(get_current_row(machine_id))
        if result["condition"] in ["Warning", "Critical"]:
            refreshed_alerts.append({
                "machine_id": result["machine_id"],
                "condition": result["condition"],
                "risk_score": result["risk_score"],
                "timestamp": result["timestamp"],
                "message": result["recommendation"],
            })
    active_alerts.clear()
    active_alerts.extend(sorted(refreshed_alerts, key=lambda x: x["risk_score"], reverse=True)[:20])

    return {
        "updated_count": len(updated),
        "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machines": updated,
        "alerts": active_alerts,
    }


@app.post("/api/machines/reset")
def reset_simulation():
    for machine_id in machine_positions:
        machine_positions[machine_id] = 0
    last_updated.clear()
    analysis_history.clear()
    active_alerts.clear()
    return {"status": "reset", "message": "Simulation returned to the first record for every machine."}


@app.get("/api/simulation/status")
def simulation_status():
    return {
        "mode": "manual",
        "update_interval_seconds": 30,
        "automatic_updates": False,
        "machines": {
            str(mid): {
                "record_index": machine_positions[mid] + 1,
                "total_records": len(machine_rows[mid]),
                "last_updated": last_updated.get(mid),
            }
            for mid in sorted(machine_rows)
        },
    }


@app.post("/api/simulate")
def simulate(request: SimulationRequest):
    base = get_current_row(request.machine_id)
    for field in ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]:
        value = getattr(request, field)
        if value is not None:
            base[field] = value
    return risk_from_row(base)


@app.post("/api/analyze")
def analyze(request: AnalysisRequest):
    result = risk_from_row(get_current_row(request.machine_id))
    analysis_history.insert(0, result)
    del analysis_history[50:]
    if result["condition"] in ["Warning", "Critical"]:
        active_alerts.insert(0, {
            "machine_id": result["machine_id"],
            "condition": result["condition"],
            "risk_score": result["risk_score"],
            "timestamp": result["timestamp"],
            "message": result["recommendation"],
        })
        del active_alerts[20:]
    return result


@app.get("/api/history")
def history():
    return analysis_history


@app.get("/api/alerts")
def alerts():
    return active_alerts


if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
