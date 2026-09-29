"""
Train & Evaluate Multi-Class ML Models for 3-Phase Induction Motor Predictive Maintenance
----------------------------------------------------------------------------------------
Models evaluated (as per project objectives):
  1. Random Forest (RF)
  2. Support Vector Machine (SVM)
  3. K-Nearest Neighbors (KNN)

Evaluates: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
Saves best model and scaler for real-time inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from dataset_generator import generate_motor_dataset

# Fault class labels mapping
FAULT_CLASSES = {
    0: "Normal / Healthy",
    1: "Bearing Degradation",
    2: "Overheating (Thermal)",
    3: "Rotor Imbalance",
    4: "Electrical Overcurrent"
}

FEATURE_COLS = [
    "temperature_c",
    "current_a",
    "rpm",
    "vib_x_g",
    "vib_y_g",
    "vib_z_g",
    "vib_total_rms"
]


def train_and_evaluate():
    print("=" * 70)
    print("STEP 1: Generating / Loading Motor Telemetry Data...")
    print("=" * 70)
    
    csv_file = "motor_telemetry_dataset.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f"Loaded existing dataset from {csv_file}")
    else:
        df = generate_motor_dataset(num_samples_per_class=2000)
        df.to_csv(csv_file, index=False)
        print(f"Generated new dataset with {len(df)} samples.")

    X = df[FEATURE_COLS]
    y = df["fault_class"]

    # Split dataset into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models dictionary
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42
        ),
        "Support Vector Machine (SVM)": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=42
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=5, weights="distance"
        )
    }

    results = {}
    best_acc = 0.0
    best_model_name = None
    best_model_obj = None

    print("\n" + "=" * 70)
    print("STEP 2: Training and Evaluating Models (RF vs SVM vs KNN)...")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "accuracy": acc,
            "predictions": y_pred,
            "model": model
        }

        print(f"--> {name} Test Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:")
        target_names = [FAULT_CLASSES[i] for i in range(len(FAULT_CLASSES))]
        print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_obj = model

    # Save artifacts
    print("\n" + "=" * 70)
    print("STEP 3: Saving Best Model & Scaler Artifacts")
    print("=" * 70)
    print(f"Top Performing Model: {best_model_name} with Accuracy = {best_acc * 100:.2f}%")

    model_save_path = "best_motor_fault_model.joblib"
    scaler_save_path = "motor_scaler.joblib"

    joblib.dump(best_model_obj, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"Model saved to: {model_save_path}")
    print(f"Scaler saved to: {scaler_save_path}")

    # Comparative Summary
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION SUMMARY (For Project Report / Presentation)")
    print("=" * 70)
    print(f"{'Model':<30} | {'Test Accuracy':<15}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<30} | {res['accuracy'] * 100:>12.2f}%")
    print("=" * 70)

    return best_model_obj, scaler


if __name__ == "__main__":
    train_and_evaluate()
