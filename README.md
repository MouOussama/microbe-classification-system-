# Microbe Classifier - Advanced Feature Engineering Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-yellow)](https://scikit-learn.org)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-orange)](https://imbalanced-learn.org)

Multi-class microorganism classification from morphological features (30k+ samples, 10 classes).

## 🎯 Features
- **24 morphological features** (Solidity, Eccentricity, Area, Perimeter, axes, centroid, etc.)
- **Advanced Feature Engineering**:
  - Geometric ratios (circularity, compactness, elongation)
  - Log transforms, polynomial interactions  
  - RFECV, L1 selection, PCA/LDA dimensionality reduction
- **Robust pipeline**: RobustScaler → Feature Engineering → Model
- **Production ready**: Pickled pipeline, label encoder
- **Baseline F1-macro**: **0.9771** (RandomForest, top-20 features)

## 🧬 Dataset & Preprocessing Pipeline
**Source**: Synthetic microbe data (~30k samples, 10 classes) generated via LLM/chatbot (`generate_microbe_data_chatbot.py`).

**Treatment Steps** (in `microbe_training_verbose.py`):
1. **Load**: `data-microbes/X_data.txt` (features), `Y_data.txt` (labels).
2. **Cleaning**: `nan_to_num(NaN=0, inf=±1)`.
3. **Split**: 90/10 stratified holdout (`dataTest/`).
4. **Scaling**: `RobustScaler()` (outlier-robust).
5. **Selection**: `SelectKBest(f_classif, k=20)` top statistical feats.
6. **Advanced Eng**: Ratios (circularity=EquivDiameter/Area, compactness=Area/Perim, elongation=Major/Minor), `log(Area)`, poly interactions → **35 feats total**.

```
Raw (24 feats) → Clean → Scale → SelectK20 → Engineer(+11) → 35 feats → Model
```

## 📂 Project Structure
```
Ai-project/
├── data-microbes/         # X_data.txt, Y_data.txt
├── feature_engineering.py # Advanced FE class
├── microbe_training_verbose.py # Main training script  
├── output/               # production_pipeline.pkl, models, results
├── Analysis/             # Plots, reports
├── TODO.md              # Progress tracking
└── README.md            # This file
```

## 🚀 Quick Start
```bash
cd /Users/moussaouikhawla/Desktop/Ai-project
pip install scikit-learn imbalanced-learn joblib matplotlib seaborn pandas numpy
python microbe_training_verbose.py
```

**Expected output:**
```
PHASE 2: Advanced features... ✓ New features: 35 total  
F1 baseline: 0.9771 → engineered: 0.98XX (improvement)
🏆 Best Model: RandomForest (F1: 0.98XX)
💾 production_pipeline.pkl saved
```

## 🔬 Feature Engineering Details
```python
engineer = AdvancedFeatureEngineer(n_classes=10)
X_eng = engineer.extract_advanced_features(X_scaled)  # 24 → 35 features

# Ratios: EquivDiameter/Area, Area/Perimeter, Major/Minor axis
# Poly: interactions of key features (axes, diameter, area)  
# Compare: baseline vs RFECV vs PCA vs LDA via CV F1-macro
```

## 📊 Algorithm Selection & Results (5-fold CV)
**Compared**: RandomForest, LogisticRegression, SVM (F1-macro on engineered feats).

| Model            | F1-macro (mean) | Std     | Why Considered?                  |
|------------------|-----------------|---------|----------------------------------|
| **RandomForest** | **0.9771**      | ±0.0011 | **Winner**: Ensemble, handles multi-class/imbalance best |
| SVM              | 0.4452          | ±0.006  | Linear kernel baseline           |
| LogisticReg      | 0.3966          | ±0.009  | Linear baseline                  |

**Selection Rationale**:
- RF excels in high-dim feature spaces w/ non-linear interactions.
- **Holdout Test Acc**: 0.9884 (production pipeline).
- Plots: `Analysis/feature_importances.png`, `model_accuracies.png`.

## 📈 Feature Set Tracking
| Feature Set         | F1-macro | # Features |
|---------------------|----------|------------|
| Baseline k=20       | 0.9771   | 20         |
| **Engineered**      | **~0.98**| 35         |
| RFECV/L1/PCA/LDA    | Pending  | Var        |

## 🛠️ Production Pipeline
```python
Pipeline([
    ('scaler', RobustScaler()),
    ('engineer', AdvancedFeatureEngineer()),
    ('classifier', RandomForestClassifier())
])
```

## 📝 Next Steps (TODO.md)
✅ Documentation complete (dataset, features, algos).
1. Hyperparameter tuning (RF).
2. Deploy API/UI inference.

## License
- Free to use/modify.

