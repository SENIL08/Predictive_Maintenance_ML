# Predictive Maintenance Framework for 3-Phase Induction Motor using Machine Learning
**Sarvajanik College of Engineering and Technology (SCET), Surat**  
*Department of Electronics and Communication Engineering (B.Tech Sem VII)*  
**Team Members:** Jay Trivedi, Vatsal Kevadiya, Senil Mangukiya, Sagar Singala  
**Mentor:** Prof. Dr. Sarosh Dastoor  

---

## 📁 Project Structure

```
predictive_maintenance_ml/
├── dataset_generator.py      # Generates / prepares calibrated motor sensor data
├── train_model.py            # Trains & compares RF, SVM, and KNN with metrics & confusion matrices
├── predict_realtime.py       # Real-time inference engine with threshold-based early warnings
├── esp32_sensor_node.ino     # Complete ESP32 Arduino firmware for MPU6050, LM35, ACS712, RPM & ThingSpeak
└── README.md                 # Setup and execution guide
```

---

## 🚀 Step-by-Step Instructions

### Step 1: Install Requirements
Make sure you have Python 3.8+ installed, then install the required libraries:
```bash
pip install numpy pandas scikit-learn joblib
```

### Step 2: Train & Evaluate the ML Models
Run the training script to train and benchmark **Random Forest**, **SVM**, and **KNN**:
```bash
python train_model.py
```
This will:
1. Load or generate the 5-class induction motor telemetry dataset (`motor_telemetry_dataset.csv`).
2. Train Random Forest (RF), Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).
3. Display comparative accuracy, precision, recall, F1-scores, and confusion matrices.
4. Save the highest-performing model (`best_motor_fault_model.joblib`) and feature normalizer (`motor_scaler.joblib`).

### Step 3: Launch the Real-Time Web Dashboard
Run the dashboard server:
```bash
python dashboard_app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

#### Dashboard Features:
1. **Live Sensor Gauges (Matching Slide 9)**:
   - Vibration RMS ($g$) with $A_x, A_y, A_z$ breakdown
   - Stator Temperature (°C)
   - Phase Current (A)
   - Motor Speed (RPM) and Slip %
2. **AI Diagnostic & Health Score**:
   - **Predicted Condition** (`NORMAL`, `BEARING DEGRADATION`, `OVERHEATING`, `ROTOR IMBALANCE`, `PHASE OVERCURRENT`)
   - **Model Confidence (%)**
   - **Health Score Ring** (100% down to 0%)
   - **Actionable Maintenance Advisory**
3. **Dual Telemetry Modes**:
   - **Live ThingSpeak Cloud Feed**: Enter your Channel ID and Read API Key in the Settings modal.
   - **Fault Injection Simulator**: Instant test buttons to simulate and demonstrate each fault state live for presentations and reviews!
4. **Real-Time Rolling Charts**:
   - Vibration RMS waveform trend
   - Temperature vs. Current correlation graph

### Step 4: Test Terminal Real-Time Inference (Optional)
Run the command-line inference script:
```bash
python predict_realtime.py
```

### Step 5: Flash ESP32 Hardware Prototype
1. Open [`esp32_sensor_node.ino`](esp32_sensor_node.ino) in Arduino IDE.
2. Install libraries:
   - `MPU6050` by Electronic Cats / Jeff Rowberg
   - `WiFi` and `HTTPClient` (built-in with ESP32 board package)
3. Set your Wi-Fi SSID, Password, and ThingSpeak Write API Key.
4. Connect pins as defined:
   - **MPU6050**: SDA -> GPIO 21, SCL -> GPIO 22
   - **LM35**: Vout -> GPIO 34 (ADC1)
   - **ACS712**: Vout -> GPIO 35 (ADC1)
   - **Rotary/Optical Encoder**: Signal -> GPIO 18 (Interrupt)
5. Flash to your ESP32 board and open Serial Monitor at 115200 baud.
