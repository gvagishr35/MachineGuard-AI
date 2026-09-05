import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


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

TARGET = "anomaly_flag"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("MachineGuard AI - Model Training")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully: {df.shape}")


# ============================================================
# TIMESTAMP PROCESSING
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort chronologically
df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[FEATURES]
y = df[TARGET]


print("\n===== FEATURES =====")
print(FEATURES)

print("\n===== TARGET =====")
print(TARGET)

print("\n===== DATASET SHAPE =====")
print("X:", X.shape)
print("y:", y.shape)


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\n===== TRAINING SET =====")
print("Rows:", len(X_train))
print("Start:", df["timestamp"].iloc[0])
print("End:", df["timestamp"].iloc[split_index - 1])

print("\n===== TEST SET =====")
print("Rows:", len(X_test))
print("Start:", df["timestamp"].iloc[split_index])
print("End:", df["timestamp"].iloc[-1])


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n===== TRAIN ANOMALY DISTRIBUTION =====")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)

print("\n===== TEST ANOMALY DISTRIBUTION =====")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True) * 100)


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING RANDOM FOREST MODEL")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n===== TRAINING MODEL =====")

model.fit(X_train, y_train)

print("Model training completed successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n===== GENERATING TEST PREDICTIONS =====")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions generated.")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


print("\n===== CONFUSION MATRIX =====")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\n===== ROC-AUC =====")

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print(round(roc_auc, 4))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

os.makedirs(
    "models",
    exist_ok=True
)


# Save the model together with the
# feature information needed by the backend.

model_package = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "model_type": "RandomForestClassifier",
    "threshold": 0.50
}


joblib.dump(
    model_package,
    MODEL_PATH
)


print("\nModel saved successfully!")

print(
    f"Model path: {MODEL_PATH}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nDataset rows: {len(df):,}")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Features: {len(FEATURES)}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nMachineGuard AI model is ready.")