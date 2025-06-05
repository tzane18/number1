import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Step 1: Load wr_features_predraft_final.parquet
print("Step 1: Loading wr_features_predraft_final.parquet...")
try:
    df = pd.read_parquet("wr_features_predraft_final.parquet")
    print("Successfully loaded wr_features_predraft_final.parquet.")
    print(f"DataFrame shape: {df.shape}")
except FileNotFoundError:
    print("Error: wr_features_predraft_final.parquet not found. Please ensure this file exists.")
    exit()
except Exception as e:
    print(f"Error loading wr_features_predraft_final.parquet: {e}")
    exit()
print("-" * 30)

# Step 2: Separate features (X) from the target variable target_AV (y)
print("\nStep 2: Separating features (X) and target (y)...")
if 'target_AV' not in df.columns:
    print("Error: 'target_AV' column not found in the DataFrame.")
    exit()

X = df.drop(columns=['target_AV'])
y = df['target_AV']
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Verify X is purely numeric and no NaNs (should be guaranteed by feature_engineering_predraft.py)
if X.isnull().sum().sum() > 0:
    print(f"WARNING: NaNs found in X after loading! Count: {X.isnull().sum().sum()}. This indicates an issue with the input parquet file.")
    # This script will not perform emergency imputation; it relies on the parquet being clean.
    # However, sklearn's StandardScaler would fail if NaNs are present.
    # For robustness of this script if previous step had issues:
    numeric_cols = X.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            print(f"Emergency imputing NaNs in {col} with median for this run.")
            X[col].fillna(X[col].median(), inplace=True)
    if X.isnull().sum().sum() > 0:
        print("ERROR: NaNs still present in X after emergency fill. Exiting.")
        exit()


# Ensure 'hof' is int if it exists (was bool, should be int after last script)
if 'hof' in X.columns and X['hof'].dtype == 'bool':
    X['hof'] = X['hof'].astype(int)
    print("'hof' column (boolean) converted to int for consistency.")
print("-" * 30)

# Step 3: Print the number of rows before dropping missing target_AV
print("\nStep 3: Handling missing target_AV values...")
print(f"Number of rows before dropping NaNs in target_AV: {X.shape[0]}")

# Step 4: Remove rows where target_AV is NaN
not_na_target_indices = y.notna()
X = X[not_na_target_indices].copy()
y = y[not_na_target_indices].copy()
print(f"Number of rows after dropping NaNs in target_AV: {X.shape[0]}")
print(f"X shape after dropping NaNs: {X.shape}, y shape after dropping NaNs: {y.shape}")
if len(y) == 0:
    print("Error: No data remaining after dropping NaNs from target_AV. Cannot proceed.")
    exit()
print("-" * 30)

# Step 5: Split the data into training (80%) and testing (20%) sets
print("\nStep 5: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("-" * 30)

# Step 6: Initialize a StandardScaler
print("\nStep 6 & 7: Scaling features...")
scaler = StandardScaler()

# Step 7: Fit the StandardScaler on X_train and transform both X_train and X_test
print(f"Fitting StandardScaler on X_train (shape: {X_train.shape})...")
X_train_scaled = scaler.fit_transform(X_train)
print("Transforming X_test...")
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")
print(f"X_train_scaled shape: {X_train_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")
print("-" * 30)

# Step 8: Train a Ridge Regression model
print("\nStep 8: Training Ridge Regression model...")
try:
    ridge_model = Ridge(random_state=42)
    ridge_model.fit(X_train_scaled, y_train)
    print("Ridge Regression model trained successfully.")
except Exception as e:
    print(f"Error training Ridge Regression model: {e}")
print("-" * 30)

# Step 9: Train a Random Forest Regressor model
print("\nStep 9: Training Random Forest Regressor model...")
try:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train) # Using X_train (unscaled, but fully imputed and numeric)
    print("Random Forest Regressor model trained successfully.")
except Exception as e:
    print(f"Error training Random Forest Regressor model: {e}")
print("-" * 30)

# Step 10: Train an XGBoost Regressor model
print("\nStep 10: Training XGBoost Regressor model...")
try:
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, early_stopping_rounds=10)
    eval_set = [(X_test, y_test)] # Using X_test as evaluation set
    xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    print("XGBoost Regressor model trained successfully.")
    print(f"Best iteration for XGBoost: {xgb_model.best_iteration}")
except Exception as e:
    print(f"Error training XGBoost Regressor model: {e}")
print("-" * 30)

# Step 11: Confirm models are trained (in memory)
print("\nStep 11: Model training confirmation...")
model_confirmation = {
    "Ridge": "Not trained" if 'ridge_model' not in locals() else "Trained",
    "RandomForest": "Not trained" if 'rf_model' not in locals() else "Trained",
    "XGBoost": "Not trained" if 'xgb_model' not in locals() else "Trained"
}
print(model_confirmation)
if 'ridge_model' in locals(): print(f"Ridge model: {ridge_model}")
if 'rf_model' in locals(): print(f"Random Forest model: {rf_model}")
if 'xgb_model' in locals(): print(f"XGBoost model: {xgb_model}")

print("\nModel training script (pre-draft features) finished.")
