

from ..ai.lstm_inference import run_lstm_cbc_inference


cbc_test_result = {
    "test_result_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "observations": [
        {"test_code": "WBC", "value_numeric": 5.6},
        {"test_code": "RBC", "value_numeric": 4.5},
        {"test_code": "HGB", "value_numeric": 14.2},
        {"test_code": "HCT", "value_numeric": 42.5},
        {"test_code": "MCV", "value_numeric": 94.4},
        {"test_code": "MCH", "value_numeric": 31.5},
        {"test_code": "MCHC", "value_numeric": 33.4},
        {"test_code": "PLT", "value_numeric": 250.0}
    ]
}

inference_output = run_lstm_cbc_inference(cbc_test_result)
print(inference_output)
