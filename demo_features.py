#!/usr/bin/env python3
"""
Demo wrapper for feature_engineering.py - Loads data and runs key methods.
Run standalone: python ui/demo_features.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering import AdvancedFeatureEngineer
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data-microbes')

print("=== Feature Engineering Demo ===")
print(f"Loading data from {DATA_DIR}")

# Load data
X = np.loadtxt(os.path.join(DATA_DIR, 'X_data.txt'))
y = np.loadtxt(os.path.join(DATA_DIR, 'Y_data.txt'), dtype=int).ravel()
print(f"Data loaded: X.shape={X.shape}, y classes={len(np.unique(y))}")

n_classes = len(np.unique(y))
engineer = AdvancedFeatureEngineer(n_classes)

print("\n1. Advanced Feature Extraction:")
X_eng = engineer.extract_advanced_features(X)
print(f"   Original: {X.shape[1]} → Engineered: {X_eng.shape[1]} features")

print("\n2. Feature Selection (RFECV):")
n_selected, rfecv = engineer.select_features_rfecv(X_eng, y)

print("\n3. PCA Demo:")
X_pca, pca, var = engineer.apply_pca(X_eng, y, n_components=10)
print(f"   Explained variance: {var:.3f}")

print("\n4. LDA Demo:")
X_lda, lda = engineer.apply_lda(X_eng, y, n_components=2)

print("\n5. CV Evaluation Comparison:")
baseline_score = engineer.evaluate_feature_set(X, y, "Raw")
eng_score = engineer.evaluate_feature_set(X_eng, y, "Engineered")
print(f"   Raw F1: {baseline_score:.4f} → Engineered F1: {eng_score:.4f}")

print("\n✅ Feature Engineering Demo Complete!")
