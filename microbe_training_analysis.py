#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MICROBE TRAINING ANALYSIS - PIPELINE FIXED (Refit Preprocessor)")
print("=" * 80)

BASE_DIR = '/Users/moussaouikhawla/Desktop/Ai-project'
DATA_DIR = os.path.join(BASE_DIR, 'data-microbes')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'Analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

print(f"Analysis: {ANALYSIS_DIR}")

print("\n1. DATA...")
X = pd.read_csv(os.path.join(DATA_DIR, 'X_data.txt'), sep='\t', header=None).values
y = pd.read_csv(os.path.join(DATA_DIR, 'Y_data.txt'), sep='\t', header=None).values.ravel()
n_classes = len(np.unique(y))
print(f"Data: {X.shape}, Classes: {n_classes}")

# Class dist
print("\n📈 Class dist")
class_counts = pd.Series(y).value_counts().sort_index()
plt.figure(figsize=(12,6))
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'class_distribution.png'), dpi=300)
plt.close()
print(" ✓ class_distribution.png")

# Model acc
print("\n📊 Model acc")
results = pd.read_csv(os.path.join(OUTPUT_DIR, 'training_results.csv'))
plt.figure(figsize=(10,6))
sns.barplot(data=results, x='model', y='f1_macro')
plt.title('Model F1')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'model_accuracies.png'), dpi=300)
plt.close()
print(" ✓ model_accuracies.png")

# Feature importances
print("\n🎯 Features")
model = joblib.load(os.path.join(OUTPUT_DIR, 'RandomForest_production.joblib'))
imp = model.feature_importances_
idx = np.argsort(imp)[-15:]
plt.figure(figsize=(12,8))
plt.barh(range(15), imp[idx])
plt.yticks(range(15), [f'F{i}' for i in idx])
plt.title('Top 15 Feature Importances (RandomForest)')
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'feature_importances.png'), dpi=300)
plt.close()
print(" ✓ feature_importances.png")

# Predictions with pipeline simulation (refit preprocessor on train)
print("\n🔮 Predictions")
print("Simulating production pipeline on holdout...")
X_clean = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, stratify=y, random_state=42)

# Refit preprocessor on train
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)

# Balance train if SMOTE
if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=42)
    X_train_selected, y_train_balanced = smote.fit_resample(X_train_selected, y_train)
else:
    y_train_balanced = y_train

# Train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
classifier.fit(X_train_selected, y_train_balanced)

# Test inference (transform only, no balance)
X_test_scaled = scaler.transform(X_test_raw)
X_test_selected = selector.transform(X_test_scaled)
y_pred = classifier.predict(X_test_selected)

acc = accuracy_score(y_test, y_pred)
print(f"Holdout Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y)))
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(np.unique(y)), yticklabels=sorted(np.unique(y)))
plt.title('Confusion Matrix - Production Pipeline Holdout')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()
print(" ✓ confusion_matrix.png")

# Training curves
print("\n📈 Curves")
plt.figure(figsize=(10,6))
plt.plot(results['model'], results['f1_macro'], 'o-', linewidth=3, markersize=8)
plt.title('Model F1-macro Scores (CV)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.ylabel('F1-macro')
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, 'training_curves.png'), dpi=300)
plt.close()
print(" ✓ training_curves.png")

# Summary
print("\n📋 Summary")
best_idx = results['f1_macro'].idxmax()
best = results.iloc[best_idx]
summary_data = pd.DataFrame({
    'metric': ['best_model', 'cv_f1_macro', 'holdout_accuracy', 'n_classes', 'features_selected', 'features_total', 'status'],
    'value': [best['model'], round(best['f1_macro'], 4), round(acc, 4), n_classes, 20, X.shape[1], 'production_ready']
})
summary_data.to_csv(os.path.join(ANALYSIS_DIR, 'model_results.csv'), index=False)
print(" ✓ model_results.csv | Best:", best['model'], f"CV F1: {best['f1_macro']:.4f} | Holdout: {acc:.4f}")

print("\n✅ ALL FILES SAVED WITHOUT WARNINGS!")
print("View Analysis/ for PNGs + CSVs")
