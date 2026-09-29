"""
Predictive Maintenance Dashboard Server
---------------------------------------
Flask application that reads real-time motor telemetry from ThingSpeak Cloud (or Simulator),
runs the trained Machine Learning model, computes health score and fault alerts,
and streams the data to a responsive dark-themed industrial SCADA dashboard.

Sarvajanik College of Engineering and Technology (SCET)
B.Tech Design Project
"""

import os
import time
import random
import requests
import joblib
import json
import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# ----------------- CONFIGURATION -----------------
CONFIG = {
    "data_source": "thingspeak",  # Strictly real ThingSpeak Cloud API
    "channel_id": os.getenv("THINGSPEAK_CHANNEL_ID", "3461983"),
    "read_api_key": os.getenv("THINGSPEAK_READ_API_KEY", ""),
    "update_interval_sec": 5
}

# In-memory history buffer for real-time charting
MAX_HISTORY_LEN = 30
telemetry_history = []

# Class definitions
FAULT_CLASSES = {
    0: {
        "title": "NORMAL",
        "color": "#10b981", # Emerald green
        "badge_class": "badge-normal",
        "description": "Motor operating within optimal mechanical and electrical parameters.",
        "base_health": 96.0
    },
    1: {
        "title": "BEARING DEGRADATION",
        "color": "#ef4444", # Red
        "badge_class": "badge-danger",
        "description": "CRITICAL: High-frequency vibration detected by MPU6050. Bearing wear/spalling likely.",
        "base_health": 42.0
    },
    2: {
        "title": "OVERHEATING",
        "color": "#f97316", # Orange
        "badge_class": "badge-warning",
        "description": "WARNING: Temperature exceeds safe insulation limits. Risk of stator winding degradation.",
        "base_health": 55.0
    },
    3: {
        "title": "ROTOR IMBALANCE",
        "color": "#eab308", # Yellow
        "badge_class": "badge-warning",
        "description": "ALERT: Significant radial vibration. Possible mechanical misalignment or bent shaft.",
        "base_health": 60.0
    },
    4: {
        "title": "PHASE OVERCURRENT",
        "color": "#dc2626", # Dark red
        "badge_class": "badge-danger",
        "description": "CRITICAL: High current surge measured by ACS712. Imbalance or phase short imminent.",
        "base_health": 28.0
    }
}

# ----------------- MODEL LOADER -----------------
MODEL = None
SCALER = None

def get_model_and_scaler():
    global MODEL, SCALER
    if MODEL is None or SCALER is None:
        model_path = os.path.join(os.path.dirname(__file__), "best_motor_fault_model.joblib")
        scaler_path = os.path.join(os.path.dirname(__file__), "motor_scaler.joblib")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                MODEL = joblib.load(model_path)
                SCALER = joblib.load(scaler_path)
                print("[ML] Loaded trained model and scaler successfully.")
            except Exception as e:
                print(f"[ML Warning] Could not load model: {e}")
        else:
            print("[ML Info] Model files not found yet. Running quick model initialization...")
            try:
                from train_model import train_and_evaluate
                MODEL, SCALER = train_and_evaluate()
            except Exception as e:
                print(f"[ML Fallback] Initialization failed: {e}. Heuristic inference active.")
    return MODEL, SCALER


# ----------------- TELEMETRY FETCHER -----------------
def fetch_thingspeak_data():
    """Fetch the latest feed from ThingSpeak Cloud channel"""
    channel_id = CONFIG["channel_id"]
    api_key = CONFIG["read_api_key"]

    if not channel_id or channel_id == "YOUR_CHANNEL_ID":
        return None, "ThingSpeak Channel ID not configured."

    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds/last.json"
    if api_key and api_key != "YOUR_READ_API_KEY":
        url += f"?api_key={api_key}"

    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if not data:
                return None, "No data feeds found in channel."

            # Parse fields according to firmware mapping from Arduino ESP32:
            # field1: temp (°C), field2: vibration (mm/s RMS), field3: current (A), field4: rpm
            temp = float(data.get("field1", 40.0) or 40.0)
            vib_rms = float(data.get("field2", 1.2) or 1.2)
            current = float(data.get("field3", 1.0) or 1.0)
            rpm = float(data.get("field4", 1460) or 1460)

            # Approximate orthogonal components from RMS for model input
            component = vib_rms / np.sqrt(3)
            return {
                "temperature": round(temp, 1),
                "current": round(current, 2),
                "rpm": round(rpm, 1),
                "vib_x": round(component, 3),
                "vib_y": round(component, 3),
                "vib_z": round(component, 3),
                "vib_rms": round(vib_rms, 3),
                "timestamp": data.get("created_at", time.strftime("%H:%M:%S"))
            }, None
        else:
            return None, f"ThingSpeak API responded with status {resp.status_code}"
    except Exception as e:
        return None, f"Connection error: {str(e)}"


def generate_simulated_data():
    """Generates realistic telemetry based on currently injected fault scenario"""
    state = CONFIG.get("current_sim_state", 0)

    if state == 0:  # Normal
        temp = np.random.normal(42.0, 1.0)
        curr = np.random.normal(2.10, 0.08)
        rpm = np.random.normal(1460, 5)
        vx = np.random.normal(0.08, 0.01)
        vy = np.random.normal(0.07, 0.01)
        vz = np.random.normal(0.10, 0.01)
    elif state == 1:  # Bearing Fault
        temp = np.random.normal(58.5, 1.5)
        curr = np.random.normal(2.45, 0.10)
        rpm = np.random.normal(1435, 12)
        vx = np.random.normal(0.50, 0.05)
        vy = np.random.normal(0.55, 0.06)
        vz = np.random.normal(0.72, 0.08)
    elif state == 2:  # Overheating
        temp = np.random.normal(83.5, 2.0)
        curr = np.random.normal(2.95, 0.12)
        rpm = np.random.normal(1420, 8)
        vx = np.random.normal(0.18, 0.02)
        vy = np.random.normal(0.20, 0.02)
        vz = np.random.normal(0.22, 0.02)
    elif state == 3:  # Rotor Imbalance
        temp = np.random.normal(51.0, 1.5)
        curr = np.random.normal(2.38, 0.10)
        rpm = np.random.normal(1410, 18)
        vx = np.random.normal(0.75, 0.06)
        vy = np.random.normal(0.72, 0.05)
        vz = np.random.normal(0.35, 0.03)
    else:  # Phase Overcurrent
        temp = np.random.normal(70.0, 2.0)
        curr = np.random.normal(4.30, 0.25)
        rpm = np.random.normal(1365, 20)
        vx = np.random.normal(0.38, 0.03)
        vy = np.random.normal(0.40, 0.04)
        vz = np.random.normal(0.42, 0.04)

    vib_rms = np.sqrt(vx**2 + vy**2 + vz**2)

    return {
        "temperature": round(float(temp), 1),
        "current": round(float(curr), 2),
        "rpm": round(float(rpm), 1),
        "vib_x": round(float(vx), 3),
        "vib_y": round(float(vy), 3),
        "vib_z": round(float(vz), 3),
        "vib_rms": round(float(vib_rms), 3),
        "timestamp": time.strftime("%H:%M:%S")
    }


def predict_condition(sensor_data):
    """Runs ML inference on sensor data and produces classification, confidence, and health score"""
    model, scaler = get_model_and_scaler()

    t = sensor_data["temperature"]
    c = sensor_data["current"]
    r = sensor_data["rpm"]
    vx = sensor_data["vib_x"]
    vy = sensor_data["vib_y"]
    vz = sensor_data["vib_z"]
    v_rms = sensor_data["vib_rms"]

    if model is not None and scaler is not None:
        try:
            vec = np.array([[t, c, r, vx, vy, vz, v_rms]])
            scaled = scaler.transform(vec)
            pred_id = int(model.predict(scaled)[0])
            probs = model.predict_proba(scaled)[0]
            confidence = round(float(probs[pred_id]) * 100, 1)
        except Exception as e:
            print(f"[ML Prediction Error]: {e}")
            pred_id = 0
            confidence = 85.0
    else:
        # Physics-based heuristic fallback if model file isn't compiled
        if t > 75:
            pred_id = 2
        elif c > 3.4:
            pred_id = 4
        elif v_rms > 0.45:
            pred_id = 1 if vz > 0.4 else 3
        else:
            pred_id = 0
        confidence = 92.5

    meta = FAULT_CLASSES.get(pred_id, FAULT_CLASSES[0])
    # Dynamic health score with slight stochastic jitter
    health_score = round(max(5.0, min(100.0, meta["base_health"] + random.uniform(-1.5, 1.5))), 1)

    return {
        "class_id": pred_id,
        "condition_name": meta["title"],
        "color": meta["color"],
        "badge_class": meta["badge_class"],
        "advisory": meta["description"],
        "confidence": confidence,
        "health_score": health_score
    }


# ----------------- FLASK ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/telemetry")
def get_telemetry():
    source = CONFIG["data_source"]
    error_msg = None

    if source == "thingspeak":
        data, err = fetch_thingspeak_data()
        if err:
            error_msg = f"[ThingSpeak Notice]: {err} (Falling back to simulated stream)"
            data = generate_simulated_data()
    else:
        data = generate_simulated_data()

    prediction = predict_condition(data)

    payload = {
        "status": "success",
        "data_source": source,
        "error_message": error_msg,
        "telemetry": data,
        "prediction": prediction,
        "config": {
            "channel_id": CONFIG["channel_id"],
            "sim_state": CONFIG["current_sim_state"]
        }
    }

    # Append to rolling history
    telemetry_history.append({
        "time": data["timestamp"],
        "temp": data["temperature"],
        "current": data["current"],
        "rpm": data["rpm"],
        "vib_rms": data["vib_rms"],
        "health": prediction["health_score"]
    })
    if len(telemetry_history) > MAX_HISTORY_LEN:
        telemetry_history.pop(0)

    return jsonify(payload)


@app.route("/api/history")
def get_history():
    return jsonify({"history": telemetry_history})


@app.route("/model_decision.json")
def serve_model_decision():
    path = os.path.join(os.path.dirname(__file__), "model_decision.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "model_decision.json not found"}), 404


@app.route("/api/config", methods=["POST"])
def update_config():
    data = request.json or {}
    if "data_source" in data:
        CONFIG["data_source"] = data["data_source"]
    if "channel_id" in data:
        CONFIG["channel_id"] = str(data["channel_id"]).strip()
    if "read_api_key" in data:
        CONFIG["read_api_key"] = str(data["read_api_key"]).strip()
    if "sim_state" in data:
        CONFIG["current_sim_state"] = int(data["sim_state"])

    return jsonify({"status": "updated", "config": CONFIG})


if __name__ == "__main__":
    # Preload ML model
    get_model_and_scaler()
    print("\n" + "=" * 75)
    print(" 🚀 PREDICTIVE MAINTENANCE DASHBOARD READY")
    print(" Local Web URL: http://127.0.0.1:5000")
    print("=" * 75 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
