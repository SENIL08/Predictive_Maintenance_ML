"""
Real-Time Predictive Maintenance Inference Engine
------------------------------------------------
Accepts telemetry readings (e.g., from ESP32 via ThingSpeak / MQTT / Serial),
computes derived vibration features, standardizes using saved scaler,
predicts the motor health state, and raises early warning alerts.
"""

import os
import joblib
import numpy as np

FAULT_LABELS = {
    0: ("NORMAL", "Motor operating within optimal mechanical and electrical limits."),
    1: ("BEARING FAULT", "CRITICAL ALERT: Excessive high-frequency vibration detected. Inspect/lubricate bearings."),
    2: ("OVERHEATING", "WARNING: Stator winding temperature exceeds safe threshold. Risk of insulation failure."),
    3: ("ROTOR IMBALANCE", "ALERT: Radial vibration indicates mechanical misalignment or unbalanced load."),
    4: ("PHASE OVERCURRENT", "DANGER: High current surge detected. Risk of stator winding burnout.")
}

FEATURE_NAMES = [
    "temperature_c",
    "current_a",
    "rpm",
    "vib_x_g",
    "vib_y_g",
    "vib_z_g",
    "vib_total_rms"
]


class MotorHealthPredictor:
    def __init__(self, model_path="best_motor_fault_model.joblib", scaler_path="motor_scaler.joblib"):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Model or scaler missing. Run 'python train_model.py' first to generate them."
            )
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, temp: float, current: float, rpm: float, vib_x: float, vib_y: float, vib_z: float):
        # Calculate derived feature: Total Vibration RMS
        vib_total_rms = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)

        input_vector = np.array([[temp, current, rpm, vib_x, vib_y, vib_z, vib_total_rms]])
        scaled_vector = self.scaler.transform(input_vector)

        # Prediction and class probabilities
        pred_class = self.model.predict(scaled_vector)[0]
        probabilities = self.model.predict_proba(scaled_vector)[0]
        confidence = probabilities[pred_class] * 100

        status_tag, advisory = FAULT_LABELS[pred_class]

        return {
            "predicted_class_id": int(pred_class),
            "status": status_tag,
            "confidence_percent": round(confidence, 2),
            "advisory_alert": advisory,
            "telemetry": {
                "temperature_c": temp,
                "current_a": current,
                "rpm": rpm,
                "vib_total_rms_g": round(vib_total_rms, 3)
            }
        }


# =====================================================================
# DEMO TEST CASES (Simulating real-time inputs from ESP32)
# =====================================================================
if __name__ == "__main__":
    predictor = MotorHealthPredictor()

    test_scenarios = [
        {"name": "Scenario 1: Normal Running Motor", "values": (43.2, 2.05, 1465, 0.08, 0.07, 0.09)},
        {"name": "Scenario 2: Bearing Wear-and-Tear", "values": (59.1, 2.40, 1430, 0.48, 0.55, 0.70)},
        {"name": "Scenario 3: Overheating / Cooling Fan Failure", "values": (84.5, 2.85, 1415, 0.19, 0.21, 0.20)},
        {"name": "Scenario 4: Misaligned Pulley / Rotor Imbalance", "values": (49.0, 2.30, 1405, 0.75, 0.71, 0.32)},
        {"name": "Scenario 5: Single Phasing / Heavy Current Overload", "values": (71.0, 4.35, 1360, 0.39, 0.41, 0.43)},
    ]

    print("=" * 80)
    print("3-PHASE INDUCTION MOTOR - LIVE PREDICTIVE MAINTENANCE MONITOR")
    print("=" * 80)

    for case in test_scenarios:
        t, c, r, vx, vy, vz = case["values"]
        res = predictor.predict(t, c, r, vx, vy, vz)
        print(f"\n[TEST] {case['name']}")
        print(f"  Inputs -> Temp: {t}°C | Current: {c}A | Speed: {r} RPM | RMS Vib: {res['telemetry']['vib_total_rms_g']}g")
        print(f"  Status: [{res['status']}] (Confidence: {res['confidence_percent']}%)")
        print(f"  Alert : {res['advisory_alert']}")
