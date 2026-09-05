import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/smart_manufacturing_data.csv"
MODEL_PATH = "models/machineguard_anomaly_model.joblib"

FEATURES = [
    "machine_id",
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption",
    "hour",
    "day_of_week",
]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MachineGuard AI",
    description="Predictive machine condition monitoring",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):
    raise RuntimeError(
        f"Dataset not found: {DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)


# Keep chronological order
df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# Time features used by ML model
df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)


# ============================================================
# LOAD ML MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found: {MODEL_PATH}. "
        "Run: python backend\\train_model.py"
    )


model_package = joblib.load(
    MODEL_PATH
)

model = model_package["model"]

MODEL_THRESHOLD = model_package.get(
    "threshold",
    0.50
)


# ============================================================
# IN-MEMORY STORAGE
# ============================================================

analysis_history = []

active_alerts = []


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalysisRequest(BaseModel):

    machine_id: int


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "status":
            "healthy",

        "dataset_rows":
            len(df),

        "machines":
            int(
                df["machine_id"].nunique()
            ),

        "model_loaded":
            True
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard():

    latest = (
        df
        .groupby("machine_id")
        .tail(1)
        .sort_values("machine_id")
    )


    return {

        "total_machines":
            int(
                df["machine_id"].nunique()
            ),

        "total_records":
            int(
                len(df)
            ),

        "anomalies":
            int(
                df["anomaly_flag"].sum()
            ),

        "maintenance_required":
            int(
                df["maintenance_required"].sum()
            ),

        "machines_with_anomaly":
            int(
                latest["anomaly_flag"].sum()
            ),

        "latest_timestamp":
            str(
                df["timestamp"].max()
            )
    }


# ============================================================
# MACHINES
# ============================================================

@app.get("/api/machines")
def machines():

    # Get latest record for every machine
    # and explicitly sort Machine 1 -> Machine 50
    latest = (
        df
        .groupby("machine_id")
        .tail(1)
        .sort_values(
            "machine_id",
            ascending=True
        )
    )


    result = []


    for _, row in latest.iterrows():

        result.append({

            "machine_id":
                int(
                    row["machine_id"]
                ),

            "temperature":
                round(
                    float(
                        row["temperature"]
                    ),
                    2
                ),

            "vibration":
                round(
                    float(
                        row["vibration"]
                    ),
                    2
                ),

            "humidity":
                round(
                    float(
                        row["humidity"]
                    ),
                    2
                ),

            "pressure":
                round(
                    float(
                        row["pressure"]
                    ),
                    2
                ),

            "energy_consumption":
                round(
                    float(
                        row["energy_consumption"]
                    ),
                    2
                ),

            "anomaly":
                int(
                    row["anomaly_flag"]
                ),

            "failure_type":
                str(
                    row["failure_type"]
                ),

            "maintenance_required":
                int(
                    row["maintenance_required"]
                ),

            "timestamp":
                str(
                    row["timestamp"]
                )
        })


    return result


# ============================================================
# MACHINE DETAILS
# ============================================================

@app.get("/api/machines/{machine_id}")
def machine_details(
    machine_id: int
):

    machine = df[
        df["machine_id"] == machine_id
    ]


    if machine.empty:

        raise HTTPException(
            status_code=404,
            detail="Machine not found"
        )


    row = machine.iloc[-1]


    return {

        "machine_id":
            machine_id,

        "temperature":
            round(
                float(
                    row["temperature"]
                ),
                2
            ),

        "vibration":
            round(
                float(
                    row["vibration"]
                ),
                2
            ),

        "humidity":
            round(
                float(
                    row["humidity"]
                ),
                2
            ),

        "pressure":
            round(
                float(
                    row["pressure"]
                ),
                2
            ),

        "energy_consumption":
            round(
                float(
                    row["energy_consumption"]
                ),
                2
            ),

        "machine_status":
            int(
                row["machine_status"]
            ),

        "anomaly_flag":
            int(
                row["anomaly_flag"]
            ),

        "failure_type":
            str(
                row["failure_type"]
            ),

        "maintenance_required":
            int(
                row["maintenance_required"]
            ),

        "timestamp":
            str(
                row["timestamp"]
            )
    }


# ============================================================
# SENSOR HISTORY
# ============================================================

@app.get("/api/machines/{machine_id}/sensor-data")
def sensor_data(
    machine_id: int,
    limit: int = 100
):

    machine = df[
        df["machine_id"] == machine_id
    ]


    if machine.empty:

        raise HTTPException(
            status_code=404,
            detail="Machine not found"
        )


    machine = machine.tail(
        limit
    )


    result = []


    for _, row in machine.iterrows():

        result.append({

            "timestamp":
                str(
                    row["timestamp"]
                ),

            "temperature":
                round(
                    float(
                        row["temperature"]
                    ),
                    2
                ),

            "vibration":
                round(
                    float(
                        row["vibration"]
                    ),
                    2
                ),

            "humidity":
                round(
                    float(
                        row["humidity"]
                    ),
                    2
                ),

            "pressure":
                round(
                    float(
                        row["pressure"]
                    ),
                    2
                ),

            "energy_consumption":
                round(
                    float(
                        row["energy_consumption"]
                    ),
                    2
                )
        })


    return result


# ============================================================
# AI ANALYSIS
# ============================================================

@app.post("/api/analyze")
def analyze(
    request: AnalysisRequest
):

    machine_id = request.machine_id


    machine = df[
        df["machine_id"] == machine_id
    ]


    if machine.empty:

        raise HTTPException(
            status_code=404,
            detail="Machine not found"
        )


    row = machine.iloc[-1]


    timestamp = pd.to_datetime(
        row["timestamp"]
    )


    # ========================================================
    # PREPARE MODEL INPUT
    # ========================================================

    input_data = pd.DataFrame([{

        "machine_id":
            int(
                row["machine_id"]
            ),

        "temperature":
            float(
                row["temperature"]
            ),

        "vibration":
            float(
                row["vibration"]
            ),

        "humidity":
            float(
                row["humidity"]
            ),

        "pressure":
            float(
                row["pressure"]
            ),

        "energy_consumption":
            float(
                row["energy_consumption"]
            ),

        "hour":
            int(
                timestamp.hour
            ),

        "day_of_week":
            int(
                timestamp.dayofweek
            )
    }])


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    probability = float(
        model.predict_proba(
            input_data
        )[0][1]
    )


    prediction = int(
        probability >= MODEL_THRESHOLD
    )


    risk_score = round(
        probability * 100,
        1
    )


    # ========================================================
    # CONDITION
    # ========================================================

    if risk_score >= 75:

        condition = "Critical"

    elif risk_score >= 50:

        condition = "Warning"

    elif risk_score >= 25:

        condition = "Watch"

    else:

        condition = "Healthy"


    # ========================================================
    # KEY FACTORS
    # ========================================================

    importance = (
        model.feature_importances_
    )


    factor_data = []


    for feature, score in zip(
        FEATURES,
        importance
    ):

        if feature in [

            "temperature",
            "vibration",
            "humidity",
            "pressure",
            "energy_consumption"

        ]:

            factor_data.append({

                "factor":
                    feature,

                "value":
                    round(
                        float(
                            row[feature]
                        ),
                        2
                    ),

                "importance":
                    round(
                        float(score),
                        4
                    )
            })


    factor_data.sort(
        key=lambda x:
            x["importance"],
        reverse=True
    )


    top_factors = (
        factor_data[:3]
    )


    # ========================================================
    # MAINTENANCE RECOMMENDATION
    # ========================================================

    if condition == "Critical":

        recommendation = (
            "Immediate inspection recommended. "
            "Check temperature and vibration "
            "conditions before continued operation."
        )

    elif condition == "Warning":

        recommendation = (
            "Schedule a maintenance inspection soon "
            "and closely monitor machine condition."
        )

    elif condition == "Watch":

        recommendation = (
            "Continue monitoring sensor trends "
            "for signs of deterioration."
        )

    else:

        recommendation = (
            "Machine condition appears normal. "
            "Continue routine monitoring."
        )


    # ========================================================
    # ANALYSIS RESULT
    # ========================================================

    result = {

        "machine_id":
            machine_id,

        "prediction":
            prediction,

        "anomaly_probability":
            round(
                probability,
                4
            ),

        "risk_score":
            risk_score,

        "condition":
            condition,

        "recommendation":
            recommendation,

        "timestamp":
            str(
                timestamp
            ),

        "key_factors":
            top_factors
    }


    # ========================================================
    # HISTORY
    # ========================================================

    analysis_history.insert(
        0,
        result
    )


    # Keep latest 50 analyses
    del analysis_history[50:]


    # ========================================================
    # ALERT
    # ========================================================

    if condition in [
        "Warning",
        "Critical"
    ]:

        active_alerts.insert(
            0,
            {

                "machine_id":
                    machine_id,

                "condition":
                    condition,

                "risk_score":
                    risk_score,

                "timestamp":
                    str(
                        timestamp
                    ),

                "message":
                    recommendation
            }
        )


        # Keep latest 20 alerts
        del active_alerts[20:]


    return result


# ============================================================
# HISTORY
# ============================================================

@app.get("/api/history")
def history():

    return analysis_history


# ============================================================
# ALERTS
# ============================================================

@app.get("/api/alerts")
def alerts():

    return active_alerts


# ============================================================
# SERVE FRONTEND
# ============================================================

if os.path.isdir("frontend"):

    app.mount(
        "/",
        StaticFiles(
            directory="frontend",
            html=True
        ),
        name="frontend"
    )