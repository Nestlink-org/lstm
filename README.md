# 🧬 CBC LSTM Autoencoder — Complete Blood Count Anomaly Detection

> **Project Type:** AI / Deep Learning
> **Model:** LSTM Autoencoder
> **Domain:** Hematology – Clinical Lab Analytics
> **Framework:** TensorFlow / Keras
> **Author:** Comphortine Siwende
> **Date:** October 2025

---

## 🩸 Overview

This project implements a **Long Short-Term Memory Autoencoder (LSTM-AE)** for analyzing **Complete Blood Count (CBC)** test patterns.
The model learns to **reconstruct normal hematology patterns** and identify **anomalous CBC readings** that may indicate potential clinical abnormalities such as inflammation, infection, or hematologic disorders.

The model is part of a larger pipeline for **AI-powered laboratory data analysis**, which aims to support automated flagging, decision support, and consistency checks across patient laboratory test results.

---

## ⚙️ Input Schema

The model receives CBC input features as a **time-series vector** representing normalized laboratory measurements from patient test records.

```json
{
  "message_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp_processed": "2025-10-04T10:30:00.000Z",
  "test_order_id": "ord-123e4567-e89b-12d3-a456-426614174003",
  "test_category": "CBC",
  "patient_id": "pat-123e4567-e89b-12d3-a456-426614174002",
  "patient_age": 35,
  "patient_gender": "M",
  "facility_id": "fac-123e4567-e89b-12d3-a456-426614174001",
  "facility_county": "Nairobi",
  "observation_feature_order": [
    "WBC",
    "RBC",
    "HGB",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
    "PLT"
  ],
  "ai_normalized_vector": [0.618, 0.406, 0.7, 0.615, 0.635, 0.833, 0.64, 0.54],
  "ai_feature_mask": [1, 1, 1, 1, 1, 1, 1, 1]
}
```

---

## 🧪 Output Schema

The model outputs a structured anomaly detection report including anomaly type, reconstruction errors, feature contribution scores, and severity levels.

```json
{
  "test_result_id": "241ee5c0-8d7d-462c-992c-bfbb3e8a08b0",
  "test_category": "wbc_differential",
  "test_panel_codes": ["NEU", "LYM", "MONO", "EOS", "BASO"],
  "anomaly_type": "test_result",
  "anomaly_score": 0.723,
  "severity_level": "medium",
  "detected_patterns_json": {
    "pattern_type": "monocyte_elevation",
    "deviation_sigma": 3.1,
    "clinical_significance": "possible_inflammation"
  },
  "feature_contributions": {
    "NEU": 0.15,
    "LYM": 0.08,
    "MONO": 0.52,
    "EOS": 0.2,
    "BASO": 0.05
  },
  "reconstruction_errors": {
    "NEU": 0.0008,
    "LYM": 0.0004,
    "MONO": 0.0012,
    "EOS": 0.0009,
    "BASO": 0.0003
  },
  "model_version": "lstm_ae_wbc_diff_v1.2.0",
  "detection_date": "2025-10-09T15:47:21.456Z",
  "related_test_ids": ["ord-123e4567-e89b-12d3-a456-426614174003"],
  "consistency_score": 0.85
}
```

---

## 📊 Dataset Description

- **Dataset Path:** `/home/comphortine/dev/servers/lstm/app/data/hematology_data.csv`
- **Rows:** 215
- **Features per record:** 9 (CBC analytes)
- **Input Shape (after sequencing):** `(215, 5, 9)`
- **Train / Validation / Test Split:**

  - Train → 150 samples
  - Validation → 32 samples
  - Test → 33 samples

Example (normalized):

| Sample |   WBC |   RBC |   HGB |   HCT |   MCV |   MCH |  MCHC |   PLT |   RDW |
| :----- | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 0      | 0.384 | 0.003 | 0.281 | 0.613 | 0.955 | 0.357 | 0.622 | 0.250 | 0.275 |
| 1      | 0.903 | 0.023 | 0.592 | 0.793 | 0.970 | 0.408 | 0.857 | 0.890 | 0.010 |
| 2      | 0.403 | 0.210 | 0.959 | 0.954 | 0.755 | 0.571 | 0.234 | 0.730 | 0.357 |

---

## 🧱 Model Architecture

| Layer Type                 | Output Shape  | Parameters |
| -------------------------- | ------------- | ---------- |
| **LSTM (Encoder)**         | (None, 5, 64) | 18,944     |
| **LSTM**                   | (None, 32)    | 12,416     |
| **RepeatVector**           | (None, 5, 32) | 0          |
| **LSTM (Decoder)**         | (None, 5, 32) | 8,320      |
| **LSTM**                   | (None, 5, 64) | 24,832     |
| **TimeDistributed(Dense)** | (None, 5, 9)  | 585        |
| **Total Parameters**       | **65,097**    | —          |

---

## 🧠 Training Details

| Parameter       | Value                     |
| --------------- | ------------------------- |
| Optimizer       | Adam                      |
| Loss Function   | Mean Squared Error (MSE)  |
| Batch Size      | 16                        |
| Epochs          | 50                        |
| Sequence Length | 5                         |
| Framework       | TensorFlow 2.17 / Keras   |
| Environment     | VS Code `.ipynb` notebook |

**Training Metrics:**

- Final Training Loss: `0.0703`
- Final Validation Loss: `0.0733`
- Convergence observed around Epoch 45–50

---

## 📈 Model Evaluation

| Metric                                               | Description                               | Value                 |
| ---------------------------------------------------- | ----------------------------------------- | --------------------- |
| **Reconstruction Error Threshold (95th percentile)** | Auto threshold to classify anomalies      | `0.09156`             |
| **Detected Anomalies**                               | Outlier CBC sequences flagged as abnormal | `2 / 33 test samples` |

A low MSE indicates well-learned reconstruction of normal CBC test trends.

---

## ⚠️ Anomaly Detection Logic

- Compute per-sample reconstruction error (MSE).
- If error > 95th percentile threshold → **Flag as anomaly**.
- Return anomaly score and severity classification:

  - **Low:** < 0.05
  - **Medium:** 0.05–0.15
  - **High:** > 0.15

Example visualization (reconstructed vs actual):

```python
plt.plot(original.flatten(), label="Original CBC")
plt.plot(reconstructed.flatten(), label="Reconstructed")
plt.title(f"Reconstruction Example | Anomaly: {is_anomaly}")
plt.legend()
```

---

## 💾 Model Files

| File                                | Description                            |
| ----------------------------------- | -------------------------------------- |
| `lstm_cbc_autoencoder.keras`        | Trained model weights and architecture |
| `cbc_scaler.pkl`                    | MinMaxScaler used during normalization |
| `hematology_data.csv`               | Clean CBC dataset                      |
| `notebooks/train_cbc_lstm_ae.ipynb` | Training notebook                      |

---

## 🚀 Deployment & Inference

You can integrate the model into a FastAPI backend for real-time CBC anomaly scoring:

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

model = load_model("models/lstm_cbc_autoencoder.keras")
scaler = joblib.load("models/cbc_scaler.pkl")

def predict_cbc_anomaly(input_vector):
    normalized = scaler.transform([input_vector])
    sequence = np.expand_dims([normalized], axis=0)
    reconstructed = model.predict(sequence)
    mse = np.mean(np.power(sequence - reconstructed, 2))
    return {"anomaly_score": float(mse), "is_anomaly": mse > 0.09156}
```

---

## 📘 Key Takeaways

- The LSTM Autoencoder effectively models **normal CBC value distributions**.
- High reconstruction error indicates **potential test inconsistencies or pathological trends**.
- The approach supports integration with **lab information systems** (LIS) or **AI lab QC engines**.

---

## 🧩 Future Work

- Integrate with **WBC differential panels** (NEU, LYM, MONO, EOS, BASO).
- Add **temporal drift analysis** to track patient changes over time.
- Deploy inference endpoint in FastAPI with versioned model monitoring.
- Extend model to other **hematology or chemistry panels**.

---

## 🧑‍💻 Author

**Comphortine Siwende**
Bachelor of Science in Information Technology
Multimedia University of Kenya
_AI and Healthcare Innovation Enthusiast_
**Director**[nestlink.dev](https://nestlink.dev)

📧 **Email:** [info@nestlink.dev](mailto:info@nestlink.dev)
🌐 **GitHub:** [github.com/nestlink](https://github.com/nestlink)

---
