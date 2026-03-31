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

## 📂 Project Structure
```
Ai-project/
├── data-microbes/         # X_data.txt, Y_data.txt
├── feature_engineering.py # Advanced FE class
├── microbe_training_verbose.py # Main training script  
├── output/               # production_pipeline.pkl, models, results
├── Analysis/             # Plots, reports
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

## 📈 Results Tracking
| Feature Set | F1-macro | # Features |
|-------------|----------|------------|
| Baseline (SelectKBest k=20) | **0.9771** | 20 |
| Engineered (+ratios/poly) | **Pending** | 35 |
| RFECV | **Pending** | ? |
| PCA/LDA | **Pending** | 10/9 |

## 🛠️ Production Pipeline
python
Pipeline([
    ('scaler', RobustScaler()),
    ('engineer', AdvancedFeatureEngineer()),
    ('classifier', RandomForestClassifier())
])


# Microbe Analysis PyQt5 Dashboard 🎛️

## Overview
Graphical interface to run all microbe ML scripts in organized order:
1. **Generate Data** (`generate_microbe_data_chatbot.py`)
2. **Train Model** (`microbe_training_verbose.py` - verbose output in console!)
3. **Feature Engineering Demo** 
4. **Analysis Plots** (`microbe_training_analysis.py`)
5. **PCA/LDA Plots** (`PCA-LDA-analysis_plots.py`)

## Setup
bash
pip install pyqt5 numpy pandas scikit-learn matplotlib seaborn joblib imbalanced-learn


## Run
```bash
cd /Users/moussaouikhawla/Desktop/Ai-project
python ui/main.py
```

## Usage
- Click tabs 1-5 in order.
- Console shows **full verbose output** (e.g., training progress, F1 scores).
- Outputs: 
  - Data: `chatbot_generated_data/`, `data-microbes/`
  - Models/PKL: `output/`
  - Plots: `Analysis/`
- Buttons disable during run to prevent overlaps.

## Progress


## 📝 Next Steps (TODO.md)
1. Full CV comparison (PCA/LDA/RFECV)
2. Update production pipeline with best FE  
3. Feature importance plots
4. Hyperparameter tuning

## License
Free to use/modify.
