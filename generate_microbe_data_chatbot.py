#!/usr/bin/env python3
"""
Pure Generative AI Data Generator - Chatbot-style text → Raw numerical data only
No preprocessing, just generation + save.
"""

import numpy as np
import pandas as pd
import os

SEED = 42
np.random.seed(SEED)

def generate_chatbot_microbe_data(n_samples=2000, n_features=50, n_classes=4):
    """
    Generate raw microbial data using chatbot-like template filling.
    Simulates LLM/chatbot generating biological profiles → numerical data.
    """
    print("🤖 Chatbot generating microbial genomic profiles...")
    
    # Chatbot "prompt templates" for different microbe types
    profiles = []
    
    for i in range(n_samples):
        class_idx = i % n_classes
        
        # LLM-like generation: biological parameters
        gc_content = np.clip(np.random.normal(45 + class_idx*5, 12), 20, 80)
        gene_density = np.random.normal(0.95 + class_idx*0.02, 0.1)
        ribosomal_rna = np.random.poisson(4 + class_idx)
        metabolic_score = np.random.normal(1.2 - class_idx*0.1, 0.3)
        virulence_factor = np.random.beta(2 + class_idx, 5)
        
        # 45 genomic features (random but class-correlated)
        genomic_features = np.random.normal(0, 1, 45) + class_idx*0.3
        
        profile = np.concatenate([
            [gc_content, gene_density, ribosomal_rna, metabolic_score, virulence_factor],
            genomic_features
        ])
        
        profiles.append(profile)
    
    X = np.array(profiles)
    # Simple balanced labels
    y = np.repeat(np.arange(n_classes), n_samples//n_classes)
    if len(y) < n_samples:
        y = np.append(y, np.random.choice(n_classes, n_samples - len(y), replace=False))
    
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y

# MAIN - Generate and save RAW data only
print("🚀 Starting pure generative AI data generation...")

X, y = generate_chatbot_microbe_data()

# Save raw data as .txt (tab-separated)
generated_dir = "chatbot_generated_data"
os.makedirs(generated_dir, exist_ok=True)

np.savetxt(os.path.join(generated_dir, 'X_data.txt'), X, delimiter='\t')
np.savetxt(os.path.join(generated_dir, 'Y_data.txt'), y.reshape(-1,1), delimiter='\t', fmt='%d')

# Save as DataFrame for inspection
df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
df['Microbe_Type'] = y
df.to_csv(os.path.join(generated_dir, 'microbe_profiles_full.txt'), sep='\t', index=False)

print(f"✅ Raw chatbot-generated data saved to {generated_dir}/")
print("- X_data.txt (features)")
print("- Y_data.txt (labels)")
print("- microbe_profiles_full.txt (DataFrame)")
print("\nReady for any processing pipeline!")
