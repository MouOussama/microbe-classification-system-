#!/usr/bin/env python3
"""
Model Testing & Predictions - Production Model Inference on Holdout
Uses fitted model with manual preprocessing for true holdout evaluation (no leak).
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import RobustScaler
from datetime import datetime

# Import feature engineering
from feature_engineering import AdvancedFeatureEngineer

print("= " * 80)
print("MODEL TESTING & PREDICTIONS - HOLD OUT INFERENCE")
print("= " * 80)

# Paths
BASE_DIR = '.'
TEST_DATA_DIR = os.path.join(BASE_DIR, 'dataTest')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test_results')
os.makedirs(TEST_DIR, exist_ok=True)

print("Holdout:", TEST_DATA_DIR)
print("Output:", OUTPUT_DIR)

# ===== LOAD FITTED MODEL & ENCODER =====
print("\nLoading production model & encoder...")
try:
    model = joblib.load(os.path.join(OUTPUT_DIR, 'RandomForest_production.joblib'))
    label_encoder = joblib.load(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    print("  ✓ model, encoder, scaler loaded")
    print("  Classes:", label_encoder.classes_)
    n_classes = len(label_encoder.classes_)
except Exception as e:
    print("Error loading:", e)
    print("Run Tab 2 first!")
    exit(1)

# ===== LOAD HOLD OUT TEST SET =====
print("\nLoading 10% holdout test set...")
X_test = pd.read_csv(os.path.join(TEST_DATA_DIR, 'X_data.txt'), sep='\t', header=None).values
y_test_str = pd.read_csv(os.path.join(TEST_DATA_DIR, 'Y_data.txt'), sep='\t', header=None).values.ravel()
y_test_enc = label_encoder.transform(y_test_str)
X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=-1.0)
print("  Holdout test:", X_test.shape[0], "samples,", n_classes, "classes")

# ===== PRODUCTION INFERENCE PREPROCESS =====
print("\nProduction preprocess (scaler fitted on train)...")
X_test_scaled = scaler.transform(X_test_clean)

engineer = AdvancedFeatureEngineer(n_classes)
X_test_engineered = engineer.extract_advanced_features(X_test_scaled)
print("  Engineered test:", X_test_engineered.shape)

# ===== PREDICT =====
print("\nPredicting...")
y_pred_enc = model.predict(X_test_engineered)
y_pred_str = label_encoder.inverse_transform(y_pred_enc)

# ===== METRICS =====
acc = accuracy_score(y_test_enc, y_pred_enc)
f1 = f1_score(y_test_enc, y_pred_enc, average='macro')
print("\nRESULTS (true holdout):")
print("  Accuracy:", round(acc, 4))
print("  F1-macro:", round(f1, 4))

print("\nReport:")
print(classification_report(y_test_enc, y_pred_enc, target_names=[str(c) for c in label_encoder.classes_], zero_division=0))

# ===== VISUALS =====
print("\nConfusion Matrix...")
cm = confusion_matrix(y_test_enc, y_pred_enc)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Holdout Test (Acc: {:.3f}, F1: {:.3f})'.format(acc, f1))
plt.ylabel('True') 
plt.xlabel('Pred')
plt.tight_layout()
plt.savefig(os.path.join(TEST_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ confusion_matrix.png")

# Samples
print("\nTop 10 Predictions:")
for i in range(min(10, len(y_test_str))):
    status = "✅" if y_test_str[i] == y_pred_str[i] else "❌"
    print("  {}: True={} → Pred={} {}".format(i, y_test_str[i], y_pred_str[i], status))

# Save
results_df = pd.DataFrame({
    'accuracy': [acc],
    'f1_macro': [f1],
    'test_size': [len(y_test_str)],
    'timestamp': [datetime.now().isoformat()]
})
results_df.to_csv(os.path.join(TEST_DIR, 'results.csv'), index=False)
print("\nAll results saved to", TEST_DIR)

print("\nHOLD OUT TEST COMPLETE - NO DATA LEAK!")
print("= " * 80)
