"""
Predictive Maintenance for 3-Phase Induction Motor
Dataset Generator & Preprocessing Pipeline
--------------------------------------------------
Simulates calibrated multi-sensor telemetry based on realistic motor physics:
  - MPU6050: 3-axis vibration (Ax, Ay, Az in g / m/s^2)
  - LM35: Stator/Bearing temperature (°C)
  - ACS712: Current sensor (A)
  - Rotary / Optical Encoder: Motor speed (RPM)

Fault Classes:
  0: Normal / Healthy Operation
  1: Bearing Degradation / Failure
  2: Overheating / Thermal Overload
  3: Rotor Imbalance / Misalignment
  4: Phase Overcurrent / Electrical Fault
"""

import numpy as np
import pandas as pd


def generate_motor_dataset(num_samples_per_class: int = 2000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    data = []

    # -------------------------------------------------------------
    # 0. HEALTHY / NORMAL STATE
    # Nominal specs: ~1440-1500 RPM, Current ~1.5 - 2.5 A, Temp ~35 - 50 °C, Low vibration
    # -------------------------------------------------------------
    for _ in range(num_samples_per_class):
        temp = np.random.normal(loc=42.0, scale=3.5)
        current = np.random.normal(loc=2.1, scale=0.15)
        rpm = np.random.normal(loc=1460, scale=12)

        vib_x = np.random.normal(loc=0.08, scale=0.03)
        vib_y = np.random.normal(loc=0.07, scale=0.03)
        vib_z = np.random.normal(loc=0.10, scale=0.04)

        data.append([temp, current, rpm, vib_x, vib_y, vib_z, 0])

    # -------------------------------------------------------------
    # 1. BEARING DEGRADATION / FAULT
    # Characteristics: Marked high-frequency vibration spikes (MPU6050),
    # slight friction temperature rise, minor speed fluctuations
    # -------------------------------------------------------------
    for _ in range(num_samples_per_class):
        temp = np.random.normal(loc=58.0, scale=4.0)
        current = np.random.normal(loc=2.45, scale=0.22)
        rpm = np.random.normal(loc=1435, scale=25)

        # High vibration spikes due to spalling/inner-outer race defect
        vib_x = np.random.normal(loc=0.45, scale=0.12)
        vib_y = np.random.normal(loc=0.52, scale=0.15)
        vib_z = np.random.normal(loc=0.68, scale=0.18)

        data.append([temp, current, rpm, vib_x, vib_y, vib_z, 1])

    # -------------------------------------------------------------
    # 2. OVERHEATING / THERMAL OVERLOAD
    # Characteristics: LM35 records high temperature (>70-90°C),
    # current elevated, vibration moderate
    # -------------------------------------------------------------
    for _ in range(num_samples_per_class):
        temp = np.random.normal(loc=82.0, scale=6.0)
        current = np.random.normal(loc=2.9, scale=0.3)
        rpm = np.random.normal(loc=1420, scale=20)

        vib_x = np.random.normal(loc=0.18, scale=0.05)
        vib_y = np.random.normal(loc=0.20, scale=0.06)
        vib_z = np.random.normal(loc=0.22, scale=0.06)

        data.append([temp, current, rpm, vib_x, vib_y, vib_z, 2])

    # -------------------------------------------------------------
    # 3. ROTOR IMBALANCE / MECHANICAL MISALIGNMENT
    # Characteristics: Strong rotational frequency vibration (esp. X & Y radial),
    # noticeable RPM fluctuation, slight current oscillation
    # -------------------------------------------------------------
    for _ in range(num_samples_per_class):
        temp = np.random.normal(loc=50.0, scale=4.5)
        current = np.random.normal(loc=2.35, scale=0.25)
        rpm = np.random.normal(loc=1410, scale=35)

        # Heavy radial vibration
        vib_x = np.random.normal(loc=0.72, scale=0.16)
        vib_y = np.random.normal(loc=0.69, scale=0.15)
        vib_z = np.random.normal(loc=0.35, scale=0.08)

        data.append([temp, current, rpm, vib_x, vib_y, vib_z, 3])

    # -------------------------------------------------------------
    # 4. PHASE OVERCURRENT / ELECTRICAL UNBALANCE
    # Characteristics: ACS712 measures surges (>3.5-5.0 A), motor hum/vibration,
    # RPM drop due to high slip/torque ripples
    # -------------------------------------------------------------
    for _ in range(num_samples_per_class):
        temp = np.random.normal(loc=68.0, scale=5.5)
        current = np.random.normal(loc=4.2, scale=0.55)
        rpm = np.random.normal(loc=1370, scale=45)

        vib_x = np.random.normal(loc=0.38, scale=0.10)
        vib_y = np.random.normal(loc=0.40, scale=0.11)
        vib_z = np.random.normal(loc=0.42, scale=0.12)

        data.append([temp, current, rpm, vib_x, vib_y, vib_z, 4])

    columns = [
        "temperature_c",
        "current_a",
        "rpm",
        "vib_x_g",
        "vib_y_g",
        "vib_z_g",
        "fault_class"
    ]
    df = pd.DataFrame(data, columns=columns)

    # Compute derived feature: Total Vibration Magnitude (RMS of 3 axes)
    df["vib_total_rms"] = np.sqrt(df["vib_x_g"]**2 + df["vib_y_g"]**2 + df["vib_z_g"]**2)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_motor_dataset(num_samples_per_class=2000)
    output_path = "motor_telemetry_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated dataset with {len(df)} samples saved to {output_path}")
    print(df.head())
