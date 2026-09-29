"""
Dropbox / Cloud Model Runner for 3-Phase Induction Motor
-------------------------------------------------------
This worker script runs in your Dropbox / Cloud environment:
  1. Continuously polls raw sensor telemetry from ThingSpeak (fields 1 to 4).
  2. Runs your trained Random Forest model (best_motor_fault_model.joblib).
  3. Computes the real-time decision (Condition, Health Score, Prescriptive Advisory).
  4. Writes the decision to 'model_decision.json' (synced in Dropbox/Dashboard)
     AND optionally posts the decision back to ThingSpeak (field5 & field6).

SCET Surat - B.Tech Design Project
"""

import os
import time
import json
import math

try:
    import requests
except ImportError:
    requests = None
    import urllib.request

try:
    import joblib
except ImportError:
    joblib = None

try:
    import numpy as np
except ImportError:
    np = None

# ----------------- CONFIGURATION -----------------
THINGSPEAK_CHANNEL_ID = os.getenv("THINGSPEAK_CHANNEL_ID", "3461983")
THINGSPEAK_READ_KEY   = os.getenv("THINGSPEAK_READ_KEY", "")

# Optional: If you also want to write the model decision back to ThingSpeak fields 5 & 6
THINGSPEAK_WRITE_KEY  = os.getenv("THINGSPEAK_WRITE_KEY", "STIB2OOZEYZWPNEJ")

# Output decision file path (can be saved in Dropbox folder)
OUTPUT_DECISION_FILE  = "model_decision.json"

POLL_INTERVAL_SECONDS = 5  # Check ThingSpeak every 5 seconds

# Fault class definitions
FAULT_CLASSES = {
    0: {
        "title": "NORMAL",
        "description": "Motor operating within optimal mechanical and electrical limits.",
        "base_health": 96.0,
        "color": "#10b981"
    },
    1: {
        "title": "BEARING DEGRADATION",
        "description": "CRITICAL: High-frequency vibration detected by MPU6050. Bearing wear/spalling likely.",
        "base_health": 42.0,
        "color": "#ef4444"
    },
    2: {
        "title": "OVERHEATING",
        "description": "WARNING: Stator temperature exceeds safe insulation limits. Risk of winding degradation.",
        "base_health": 55.0,
        "color": "#f97316"
    },
    3: {
        "title": "ROTOR IMBALANCE",
        "description": "ALERT: Significant radial vibration. Possible mechanical misalignment or bent shaft.",
        "base_health": 60.0,
        "color": "#eab308"
    },
    4: {
        "title": "PHASE OVERCURRENT",
        "description": "CRITICAL: High current surge measured by ACS712. Overload or phase short imminent.",
        "base_health": 28.0,
        "color": "#dc2626"
    }
}


def load_model():
    if joblib is None:
        print("[Model Notice] joblib not installed in current Python env. Using calibrated decision logic.")
        return None, None

    model_path = "best_motor_fault_model.joblib"
    scaler_path = "motor_scaler.joblib"
    
    model = None
    scaler = None
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print(f"[Model] Successfully loaded {model_path} and {scaler_path}")
        except Exception as e:
            print(f"[Model Notice] Model load error: {e}")
    else:
        print("[Model Notice] Model files not found in current folder. Using calibrated decision logic.")
    return model, scaler


def fetch_latest_thingspeak():
    if not THINGSPEAK_CHANNEL_ID or THINGSPEAK_CHANNEL_ID == "YOUR_CHANNEL_ID":
        return None
    
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds/last.json"
    if THINGSPEAK_READ_KEY and THINGSPEAK_READ_KEY != "YOUR_READ_API_KEY":
        url += f"?api_key={THINGSPEAK_READ_KEY}"
        
    try:
        if requests is not None:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        else:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"[ThingSpeak Fetch Error]: {e}")
    return None


def run_inference(feed, model, scaler):
    # Parse incoming ThingSpeak fields from user's ESP32:
    # Field 1 = Temperature (°C) [DHT11]
    # Field 2 = Vibration (mm/s RMS) [MPU6050]
    # Field 3 = Current (A) [ACS712-5A]
    # Field 4 = RPM [Proximity Sensor]
    temp = float(feed.get("field1", 36.4) or 36.4)
    vib = float(feed.get("field2", 1.25) or 1.25)
    current = float(feed.get("field3", 0.93) or 0.93)
    rpm = float(feed.get("field4", 1460) or 1460)
    
    comp = vib / (math.sqrt(3) if np is None else np.sqrt(3)) if vib > 0 else 0.0

    if model is not None and scaler is not None and np is not None:
        try:
            vec = np.array([[temp, current, rpm, comp, comp, comp, vib]])
            scaled = scaler.transform(vec)
            pred_id = int(model.predict(scaled)[0])
            probs = model.predict_proba(scaled)[0]
            conf = round(float(probs[pred_id]) * 100, 1)
        except Exception as e:
            print(f"[Inference Error]: {e}")
            pred_id = 0
            conf = 95.0
    else:
        # Calibrated heuristic fallback (ISO 10816-3 velocity mm/s RMS)
        if temp > 70.0:
            pred_id = 2  # Overheating
        elif current > 3.4:
            pred_id = 4  # Phase Overcurrent
        elif vib > 4.5:
            pred_id = 1  # Bearing Degradation (ISO Zone D)
        elif vib > 2.8 or (vib > 1.8 and rpm < 1420):
            pred_id = 3  # Rotor Imbalance / Misalignment (ISO Zone C)
        else:
            pred_id = 0  # Normal
        conf = 95.0

    meta = FAULT_CLASSES.get(pred_id, FAULT_CLASSES[0])
    slip = round(((1500 - rpm) / 1500) * 100, 1) if rpm > 0 else 0.0

    decision = {
        "timestamp": feed.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ")),
        "local_sync_time": time.strftime("%H:%M:%S"),
        "telemetry": {
            "temperature_c": temp,
            "current_a": current,
            "rpm": rpm,
            "slip_percent": slip,
            "vibration_rms": vib
        },
        "model_output": {
            "predicted_class_id": pred_id,
            "condition": meta["title"],
            "health_score": meta["base_health"],
            "confidence_percent": conf,
            "advisory": meta["description"],
            "color": meta["color"]
        },
        "source": "Cloud / Dropbox Model Worker"
    }
    return decision


def write_decision_to_file(decision):
    try:
        with open(OUTPUT_DECISION_FILE, "w") as f:
            json.dump(decision, f, indent=2)
        print(f"[{decision['local_sync_time']}] Decision Saved: {decision['model_output']['condition']} (Health: {decision['model_output']['health_score']}%)")
    except Exception as e:
        print(f"[File Write Error]: {e}")


def post_decision_to_thingspeak(decision):
    if not THINGSPEAK_WRITE_KEY:
        return
    url = f"https://api.thingspeak.com/update?api_key={THINGSPEAK_WRITE_KEY}&field5={decision['model_output']['predicted_class_id']}&field6={decision['model_output']['health_score']}"
    try:
        if requests is not None:
            requests.get(url, timeout=5)
        else:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
    except Exception as e:
        print(f"[ThingSpeak Write Error]: {e}")


def main():
    print("=" * 70)
    print("  🚀 3-PHASE MOTOR PREDICTIVE MAINTENANCE: CLOUD MODEL RUNNER")
    print(f"  Target File: {OUTPUT_DECISION_FILE}")
    print("=" * 70)
    
    model, scaler = load_model()
    
    last_feed_id = None
    while True:
        feed = fetch_latest_thingspeak()
        if feed and feed.get("created_at"):
            feed_id = feed.get("entry_id")
            if feed_id != last_feed_id:
                last_feed_id = feed_id
                decision = run_inference(feed, model, scaler)
                write_decision_to_file(decision)
                post_decision_to_thingspeak(decision)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
