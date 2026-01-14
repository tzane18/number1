"""
Train and save models for 2026 WR prediction
This script trains Ridge, Random Forest, and XGBoost models and saves them for later use
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle

print("="*60)
print("Training Models for 2026 WR Success Prediction")
print("="*60)

# Load the training data
print("\n[1/6] Loading training data...")
df = pd.read_parquet("wr_features_predraft_final.parquet")
print(f"   Loaded {df.shape[0]} records with {df.shape[1]} features")

# Separate features and target
print("\n[2/6] Preparing features and target...")
X = df.drop(columns=['target_AV'])
y = df['target_AV']

# Remove rows with missing target
not_na_indices = y.notna()
X = X[not_na_indices].copy()
y = y[not_na_indices].copy()
print(f"   Training samples: {len(X)}")
print(f"   Feature columns: {X.shape[1]}")

# Ensure hof is int
if 'hof' in X.columns and X['hof'].dtype == 'bool':
    X['hof'] = X['hof'].astype(int)

# Split data
print("\n[3/6] Splitting data (80/20 train/test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# Train models
print("\n[4/6] Training models...")

# Ridge Regression (requires scaling)
print("   - Training Ridge Regression...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train_scaled, y_train)
ridge_score = ridge_model.score(X_test_scaled, y_test)
print(f"     R² Score: {ridge_score:.3f}")

# Random Forest
print("   - Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)
print(f"     R² Score: {rf_score:.3f}")

# XGBoost
print("   - Training XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, early_stopping_rounds=10)
eval_set = [(X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
xgb_score = xgb_model.score(X_test, y_test)
print(f"     R² Score: {xgb_score:.3f}")

# Save models
print("\n[5/6] Saving models...")
with open('ridge_model.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)
print("   ✓ Saved ridge_model.pkl")

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("   ✓ Saved rf_model.pkl")

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("   ✓ Saved xgb_model.pkl")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Saved scaler.pkl")

# Save feature columns for later use
feature_columns = X_train.columns.tolist()
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
print("   ✓ Saved feature_columns.pkl")

print("\n[6/6] Model Summary:")
print(f"   Ridge Regression R²: {ridge_score:.3f}")
print(f"   Random Forest R²:    {rf_score:.3f}")
print(f"   XGBoost R²:          {xgb_score:.3f}")
print(f"   Best Model:          {'Ridge' if ridge_score >= max(rf_score, xgb_score) else ('RF' if rf_score >= xgb_score else 'XGBoost')}")

print("\n" + "="*60)
print("Model Training Complete!")
print("="*60)
