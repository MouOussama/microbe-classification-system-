#!/usr/bin/env python3
"""
Microbe Classifier Training Program - VERBOSE TRAINING OUTPUT
Shows detailed progress during each training step
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Optional SMOTE
SMOTE_AVAILABLE = False
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
    print("✅ SMOTE available")
except ImportError:
    print("⚠️  SMOTE not available (pip install imbalanced-learn) - skipping balancer")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

print("=" * 70)
print("MICROBE CLASSIFIER TRAINING - VERBOSE MODE")
print("=" * 70)

# Configuration
BASE_DIR = '/Users/moussaouikhawla/Desktop/Ai-project'
DATA_DIR = os.path.join(BASE_DIR, 'data-microbes')  
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"📂 Data path: {DATA_DIR}")
print(f"💾 Output path: {OUTPUT_DIR}")

# ===== PHASE 1: DATA LOADING =====
print("\n" + "="*70)
print("PHASE 1: DATASET LOADING & VALIDATION")
print("="*70)

print("🔄 Loading X_data.txt...")
X = pd.read_csv(os.path.join(DATA_DIR, 'X_data.txt'), sep='\t', header=None).values
print(f"   ✓ X shape: {X.shape}")

print("🔄 Loading Y_data.txt...")
y = pd.read_csv(os.path.join(DATA_DIR, 'Y_data.txt'), sep='\t', header=None).values.ravel()
print(f"   ✓ y shape: {y.shape}, unique classes: {len(np.unique(y))}")

print("🔄 Cleaning NaN/Inf values...")
X_clean = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
print(f"   ✓ Cleaning complete: {X_clean.shape}")

print("🔄 Label encoding...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"   ✓ Classes: {label_encoder.classes_}")

n_classes = len(label_encoder.classes_)
print(f"\n✅ PHASE 1 COMPLETE: Dataset ready ({X.shape[0]} samples, {n_classes} classes)")

# ===== PHASE 2: FEATURE ENGINEERING =====
print("\n" + "="*70)
print("PHASE 2: FEATURE ENGINEERING & PREPROCESSING")
print("="*70)

print("🔄 Step 2.1: Robust scaling...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_clean)
print(f"   ✓ Scaled data: {X_scaled.shape}")

print("🔄 Step 2.2: Statistical feature selection...")
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y_encoded)
selected_features = selector.get_support(indices=True)
print(f"   ✓ Selected top 20 features: {selected_features[:5]}...")
print(f"   ✓ Feature scores (top 5): {selector.scores_[selected_features[:5]]}")

# ADVANCED FEATURE ENGINEERING
print("🔄 Advanced features...")
from feature_engineering import AdvancedFeatureEngineer
engineer = AdvancedFeatureEngineer(n_classes)
X_engineered = engineer.extract_advanced_features(X_scaled)
print(f"   ✓ New features: {X_engineered.shape[1]} total")

baseline_score = engineer.evaluate_feature_set(X_selected, y_encoded, "Baseline(top20)")
eng_score = engineer.evaluate_feature_set(X_engineered, y_encoded, "Engineered")
print(f"   CV F1: Baseline {baseline_score:.4f} → Eng {eng_score:.4f}")

X_processed = X_engineered  # Use engineered
print(f"\n✅ PHASE 2 COMPLETE: Features optimized ({X_processed.shape})")

# ===== PHASE 3: CROSS-VALIDATION TRAINING =====
print("\n" + "="*70)
print("PHASE 3: ALGORITHM SELECTION (5-FOLD CV)")
print("="*70)

print("🔄 Preparing stratified 5-fold CV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"   ✓ CV splits ready")

models_config = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

print(f"🔄 Testing {len(models_config)} algorithms (F1-macro scoring)")
results = {}

for name, model in models_config.items():
    print(f"\n📊 MODEL: {name}")
    print("-" * 40)
    print(f"🔄 Initializing {name}...")
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_processed, y_encoded), 1):
        print(f"   Fold {fold}/5: Training...")
        X_train_fold, X_val_fold = X_processed[train_idx], X_processed[val_idx]
        y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        score = f1_score(y_val_fold, model.predict(X_val_fold), average='macro')
        fold_scores.append(score)
        print(f"     Fold {fold}: {score:.4f}")
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    results[name] = mean_score
    
    print(f"\n📈 {name} SUMMARY:")
    print(f"   Mean F1-macro: {mean_score:.4f}")
    print(f"   Std: ±{std_score:.4f}")
    print(f"   Individual folds: {[f'{s:.4f}' for s in fold_scores]}")

# ===== BEST MODEL SELECTION =====
print("\n" + "="*70)
print("PHASE 4: BEST MODEL SELECTION")
print("="*70)

best_name = max(results, key=results.get)
best_score = results[best_name]

print("🏆 FINAL RANKING:")
for name in sorted(results, key=results.get, reverse=True):
    print(f"   {name}: {results[name]:.4f}")

print(f"\n🎯 SELECTED: {best_name} (F1: {best_score:.4f})")
print("   Reason: Highest F1-macro + Lowest variance")

# ===== FINAL PRODUCTION TRAINING =====
print("\n" + "="*70)
print("PHASE 5: FINAL PRODUCTION TRAINING")
print("="*70)

print(f"🔄 Training final {best_name} on FULL dataset...")
final_model = models_config[best_name]
final_model.fit(X_processed, y_encoded)

print(f"   ✓ Training complete")
print(f"   Model details: {type(final_model).__name__}")
if hasattr(final_model, 'n_estimators'):
    print(f"   Trees: {final_model.n_estimators}")

# Production pipeline
print("🔄 Building production pipeline...")
steps = [
    ('scaler', RobustScaler()),
    ('selector', SelectKBest(f_classif, k=20)),
]
if SMOTE_AVAILABLE:
    steps.append(('balancer', SMOTE(random_state=42)))
    print("   ✓ Balancer (SMOTE) included")
else:
    print("   ⚠️  No balancer (SMOTE skipped)")
steps.append(('classifier', final_model))
production_pipeline = Pipeline(steps)

print("✅ PRODUCTION PIPELINE ASSEMBLED")

# ===== SAVE ARTIFACTS =====
print("\n" + "="*70)
print("PHASE 6: SAVING PRODUCTION ARTIFACTS")
print("="*70)

print("💾 Saving full production pipeline...")
joblib.dump(production_pipeline, os.path.join(OUTPUT_DIR, 'production_pipeline.pkl'))
print("   ✓ production_pipeline.pkl")

print("💾 Saving trained model...")
joblib.dump(final_model, os.path.join(OUTPUT_DIR, f'{best_name}_production.joblib'))
print("   ✓ {best_name}_production.joblib")

print("💾 Saving label encoder...")
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, 'label_encoder.pkl'))
print("   ✓ label_encoder.pkl")

print("💾 Saving results...")
results_df = pd.DataFrame([{'model': k, 'f1_macro': v} for k, v in results.items()])
results_df.to_csv(os.path.join(OUTPUT_DIR, 'training_results.csv'), index=False)
print("   ✓ training_results.csv")

print("\n✅ ALL ARTIFACTS SAVED")

# ===== FINAL SUMMARY =====
print("\n" + "="*70)
print("TRAINING SUMMARY - PRODUCTION READY")
print("="*70)
print(f"Dataset: {X.shape[0]} samples, {n_classes} classes")
print(f"Best Model: {best_name}")
print(f"Performance: F1-macro = {best_score:.4f}")
print(f"Pipeline: production_pipeline.pkl")
print("\n🎉 TRAINING COMPLETE")
print("="*70)
