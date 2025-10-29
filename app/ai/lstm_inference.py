import tensorflow as tf
import numpy as np
import uuid
from datetime import datetime, timezone

# -----------------------------------
# CBC feature order and reference normalization ranges
# -----------------------------------
CBC_FEATURES = ["WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT"]

CBC_REF_RANGES = {
    "WBC": (4.0, 11.0),
    "RBC": (4.20, 5.80),
    "HGB": (13.0, 17.0),
    "HCT": (37.0, 50.0),
    "MCV": (80.0, 100.0),
    "MCH": (27.0, 33.0),
    "MCHC": (31.0, 36.0),
    "PLT": (150.0, 400.0),
}

MODEL_VERSION = "lstm_ae_wbc_diff_v1.2.0"
MODEL_PATH = "/home/comphortine/dev/servers/lstm/app/models/lstm_cbc_autoencoder.keras"

# Load the trained model once
lstm_autoencoder = tf.keras.models.load_model(MODEL_PATH)


# -----------------------------------
# Helper Functions
# -----------------------------------

def normalize_feature(value, feature):
    """Normalize value based on reference range (min-max normalization)."""
    ref_min, ref_max = CBC_REF_RANGES.get(feature, (0, 1))
    return (value - ref_min) / (ref_max - ref_min)


def prepare_lstm_input(test_result_json: dict):
    """Extract and normalize CBC analyte values into a model-ready input vector."""
    obs = test_result_json["observations"]
    feature_vector = []
    feature_mask = []

    for feature in CBC_FEATURES:
        val = next((float(o["value_numeric"]) for o in obs if o["test_code"] == feature), None)
        if val is not None:
            feature_vector.append(normalize_feature(val, feature))
            feature_mask.append(1)
        else:
            feature_vector.append(0.0)
            feature_mask.append(0)

    input_data = {
        "message_id": str(uuid.uuid4()),
        "timestamp_processed": datetime.now(timezone.utc).isoformat(),
        "test_order_id": test_result_json["test_result_id"],
        "test_category": "CBC",
        "patient_id": "pat-unknown",
        "patient_age": None,
        "patient_gender": None,
        "facility_id": None,
        "facility_county": None,
        "observation_feature_order": CBC_FEATURES,
        "ai_normalized_vector": feature_vector,
        "ai_feature_mask": feature_mask
    }
    return input_data


# -----------------------------------
# Main Inference Function
# -----------------------------------

def run_lstm_cbc_inference(test_result_json: dict):
    """Takes CBC test result → runs LSTM Autoencoder → returns structured anomaly output."""

    input_data = prepare_lstm_input(test_result_json)
   # shape = (batch_size=1, timesteps=1, features=9)
    x_input = np.tile(np.array(input_data["ai_normalized_vector"]), (5, 1))  # repeat 5 timesteps
    x_input = x_input.reshape((1, 5, len(CBC_FEATURES)))  # batch=1, timesteps=5, features=8



    reconstructed = lstm_autoencoder.predict(x_input)
    reconstruction_error = float(np.mean(np.square(x_input - reconstructed)))

    feature_errors = {
    feature: float(np.mean(np.square(x_input[:, :, i] - reconstructed[:, :, i])))
    for i, feature in enumerate(CBC_FEATURES)
        }


    # Compute anomaly score and severity
    anomaly_score = float(min(1.0, reconstruction_error * 10))
    severity = (
        "low" if anomaly_score < 0.4
        else "medium" if anomaly_score < 0.7
        else "high"
    )

    # Generate dummy WBC differential inference (can later integrate actual WBC diff model)
    feature_contribs = {"NEU": 0.15, "LYM": 0.08, "MONO": 0.52, "EOS": 0.20, "BASO": 0.05}

    detected_pattern = {
        "pattern_type": "monocyte_elevation" if anomaly_score > 0.7 else "normal_distribution",
        "deviation_sigma": round(anomaly_score * 4, 2),
        "clinical_significance": "possible_inflammation" if anomaly_score > 0.7 else "within normal limits"
    }

    result = {
        "test_result_id": test_result_json["test_result_id"],
        "test_category": "wbc_differential",
        "test_panel_codes": list(feature_contribs.keys()),
        "anomaly_type": "test_result",
        "anomaly_score": anomaly_score,
        "severity_level": severity,
        "detected_patterns_json": detected_pattern,
        "feature_contributions": feature_contribs,
        "reconstruction_errors": feature_errors,
        "model_version": MODEL_VERSION,
        "detection_date": datetime.now(timezone.utc).isoformat(),
        "related_test_ids": [test_result_json["test_result_id"]],
        "consistency_score": round(1 - anomaly_score / 2, 2),
    }

    return result
