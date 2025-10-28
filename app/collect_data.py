import pandas as pd
import os
import uuid
from datetime import datetime, timedelta
import random

CSV_PATH = "/home/comphortine/dev/servers/lstm/app/data/hematology_data.csv"

def generate_sample_data(num_records=50):
    """Generate simulated CBC records with realistic variations."""
    base_time = datetime(2025, 10, 4, 10, 30, 0)
    facilities = ["fac-01", "fac-02", "fac-03"]
    genders = ["M", "F"]
    data = []

    for i in range(num_records):
        record = {
            "message_id": str(uuid.uuid4()),
            "test_order_id": f"ord-{str(uuid.uuid4())[:8]}",
            "patient_id": f"pat-{str(uuid.uuid4())[:8]}",
            "patient_age": random.randint(18, 70),
            "patient_gender": random.choice(genders),
            "facility_id": random.choice(facilities),
            "timestamp_processed": (base_time + timedelta(days=i)).isoformat() + "Z",
            # Random but realistic normalized CBC values (0–1)
            "WBC": round(random.uniform(0.5, 0.8), 3),
            "RBC": round(random.uniform(0.35, 0.55), 3),
            "HGB": round(random.uniform(0.65, 0.85), 3),
            "HCT": round(random.uniform(0.55, 0.75), 3),
            "MCV": round(random.uniform(0.6, 0.7), 3),
            "MCH": round(random.uniform(0.8, 0.9), 3),
            "MCHC": round(random.uniform(0.6, 0.7), 3),
            "PLT": round(random.uniform(0.5, 0.6), 3)
        }
        data.append(record)

    return data


def save_json_to_csv(json_data, csv_filename=CSV_PATH):
    """Save list of dictionaries (JSON-like) to a CSV file."""
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df = pd.DataFrame(json_data)

    if os.path.exists(csv_filename):
        df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_filename, index=False)

    print(f"✅ {len(json_data)} records saved to: {csv_filename}")


if __name__ == "__main__":
    sample_data = generate_sample_data(50)
    save_json_to_csv(sample_data)
