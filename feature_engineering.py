#!/usr/bin/env python3
"""Advanced Feature Engineering for Microbe Classification"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    def extract_advanced_features(self, X):
        """Create new engineered features: ratios, shape indices, polynomial""" 
        X_eng = X.copy()
        
        # Geometric ratios (domain knowledge for microbes)
        X_eng = np.column_stack([
            X_eng,
            X[:, 2] / (X[:, 22] + 1e-8),  # EquivDiameter / Area (circularity)
            X[:, 22] / (X[:, 18] + 1e-8),  # Area / Perimeter (compactness)
            X[:, 16] / X[:, 17],           # Major/Minor axis (elongation)
            X[:, 20] / X[:, 21],           # Centroid ratios (position)
            np.log1p(X[:, 22]),            # log(Area)
            X[:, 1] * X[:, 6],             # Eccentricity * Orientation
        ])
        
        # Clean NaN/inf from ratios
        X_eng = np.nan_to_num(X_eng, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Polynomial features (degree 2, interaction only for key features)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(X[:, [2, 16, 17, 22]])  # Diameter, axes, area
        X_eng = np.column_stack([X_eng, poly_features[:, 1:]])  # Skip constant
        X_eng = np.nan_to_num(X_eng, nan=0.0, posinf=10.0, neginf=-10.0)  # Final clean
        
        print(f"Engineered features: {X_eng.shape[1]} total")
        return X_eng
    
    def select_features_rfecv(self, X, y):
        """Recursive Feature Elimination with CV"""
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rfecv = RFECV(estimator=rf, step=0.1, cv=self.cv, scoring='f1_macro', n_jobs=-1)
        rfecv.fit(X, y)
        print(f"RFECV selected: {rfecv.n_features_} features")
        return rfecv.n_features_, rfecv
    
    def select_features_l1(self, X, y):
        """L1 regularization selection"""
        lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
        selector = SelectFromModel(lr, max_features=20)
        selector.fit(X, y)
        print(f"L1 selected: {sum(selector.get_support())} features")
        return selector
    
    def apply_pca(self, X, y, n_components=10):
        """PCA dimensionality reduction"""
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"PCA: {n_components} components, explained variance: {explained_var:.3f}")
        return X_pca, pca, explained_var
    
    def apply_lda(self, X, y, n_components=None):
        """LDA (supervised, max n_classes-1)"""
        n_comp = min(n_components or self.n_classes-1, self.n_classes-1, X.shape[1]-1)
        lda = LDA(n_components=n_comp)
        X_lda = lda.fit_transform(X, y)
        print(f"LDA: {n_comp} components")
        return X_lda, lda
    
    def evaluate_feature_set(self, X, y, name):
        """CV evaluation for feature set"""
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X, y, cv=self.cv, scoring='f1_macro')
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        return scores.mean()

# Usage example:
# engineer = AdvancedFeatureEngineer(n_classes=7)
# X_eng = engineer.extract_advanced_features(X_scaled)
# score_eng = engineer.evaluate_feature_set(X_eng, y_encoded, 'Engineered')

