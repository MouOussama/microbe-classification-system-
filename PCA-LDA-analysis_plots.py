#!/usr/bin/env python3
#\"\"\"Feature Analysis Plots for Microbe Classifier\"\"\"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from feature_engineering import AdvancedFeatureEngineer
import joblib

BASE_DIR = '/Users/moussaouikhawla/Desktop/Ai-project'
ANALYSIS_DIR = os.path.join(BASE_DIR, 'Analysis')
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def generate_feature_plots(X_scaled, y_encoded, engineer):
   # \"\"\"Generate PCA variance, LDA scatter, feature rankings\"\"\"

    # 1. PCA Explained Variance
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-', linewidth=2)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance Ratio')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'pca_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved: pca_variance.png')

    # 2. LDA 2D Scatter
    lda = engineer.apply_lda(X_scaled, y_encoded, n_components=2)[1]
    X_lda = lda.fit_transform(X_scaled, y_encoded)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y_encoded, cmap='tab10', alpha=0.7)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title('LDA 2D Projection (10 Classes)')
    plt.colorbar(scatter)
    plt.savefig(os.path.join(ANALYSIS_DIR, 'lda_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved: lda_scatter.png')

    # 3. Feature Selection Ranking (RFECV)
    _, rfecv = engineer.select_features_rfecv(X_scaled, y_encoded)
    plt.figure(figsize=(12, 6))
    ranking = rfecv.ranking_
    plt.bar(range(len(ranking)), ranking)
    plt.xlabel('Feature Index')
    plt.ylabel('RFECV Ranking')
    plt.title('Feature Selection Ranking')
    plt.savefig(os.path.join(ANALYSIS_DIR, 'feature_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved: feature_ranking.png')

    print('📊 All analysis plots saved to Analysis/')

if __name__ == '__main__':
    # Load data for demo
    print('Loading data for plots...')
    X = np.loadtxt(os.path.join(BASE_DIR, 'data-microbes/X_data.txt'))
    y = np.loadtxt(os.path.join(BASE_DIR, 'data-microbes/Y_data.txt'), dtype=int)
    
    label_encoder = joblib.load(os.path.join(BASE_DIR, 'output/label_encoder.pkl'))
    X_scaled = RobustScaler().fit_transform(X)
    y_encoded = label_encoder.transform(y.ravel())
    engineer = AdvancedFeatureEngineer(len(np.unique(y_encoded)))
    
    generate_feature_plots(X_scaled, y_encoded, engineer)

