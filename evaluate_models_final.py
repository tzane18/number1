import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load wr_features_final.parquet
print("Step 1: Loading wr_features_final.parquet...")
try:
    df = pd.read_parquet("wr_features_final.parquet")
    print("Successfully loaded wr_features_final.parquet.")
    print(f"DataFrame shape: {df.shape}")
except FileNotFoundError:
    print("Error: wr_features_final.parquet not found. Please ensure this file exists.")
    exit()
except Exception as e:
    print(f"Error loading wr_features_final.parquet: {e}")
    exit()
print("-" * 30)

# Step 2: Separate features (X) from the target variable target_AV (y)
print("\nStep 2: Separating features (X) and target (y)...")
if 'target_AV' not in df.columns:
    print("Error: 'target_AV' column not found in the DataFrame.")
    exit()

X = df.drop(columns=['target_AV'])
y = df['target_AV']

# Store original column names from X (after potential minor preprocessing like 'hof' conversion)
# Ensure 'hof' is int, as it might be boolean from parquet
if 'hof' in X.columns and X['hof'].dtype == 'bool':
    X['hof'] = X['hof'].astype(int)
    print("'hof' column converted to int type for consistency.")
X_column_names = X.columns.tolist() # Store after 'hof' conversion
print(f"X shape: {X.shape}, y shape: {y.shape}. Stored {len(X_column_names)} feature names.")

# Verify no NaNs in X (should be handled by feature_engineering_final.py)
if X.isnull().sum().sum() > 0:
    print(f"WARNING: NaNs found in X after loading! Count: {X.isnull().sum().sum()}. This should have been handled in the previous step.")
    # Fallback: Impute with median for any numeric columns still containing NaNs
    numeric_cols_in_X = X.select_dtypes(include=np.number).columns
    for col in numeric_cols_in_X:
        if X[col].isnull().any():
            print(f"Emergency imputing NaNs in {col} with median.")
            X[col].fillna(X[col].median(), inplace=True)
print("-" * 30)


# Step 3: Print the number of rows before dropping missing target_AV
print("\nStep 3: Handling missing target_AV values...")
print(f"Number of rows before dropping NaNs in target_AV: {X.shape[0]}")

# Step 4: Remove rows where target_AV is NaN
not_na_target_indices = y.notna()
X = X[not_na_target_indices].copy()
y = y[not_na_target_indices].copy()
print(f"Number of rows after dropping NaNs in target_AV: {X.shape[0]}")
print(f"X shape after filtering for y: {X.shape}, y shape: {y.shape}")
if len(y) == 0:
    print("Error: No data remaining after dropping NaNs from target_AV. Cannot proceed.")
    exit()
print("-" * 30)

# Step 5: Split data into training (80%) and testing (20%) sets
print("\nStep 5: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
# Ensure X_column_names matches the number of features in X_train and X_test
if len(X_column_names) != X_train.shape[1]:
    print(f"Warning: Column name list length ({len(X_column_names)}) does not match X_train feature count ({X_train.shape[1]}). Resetting column names from X_train.")
    X_column_names = X_train.columns.tolist()
print("-" * 30)

# Step 6: Initialize StandardScaler
print("\nStep 6 & 7: Scaling features...")
scaler = StandardScaler()

# Step 7: Fit StandardScaler on X_train. Transform X_train and X_test.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")
print(f"X_train_scaled shape: {X_train_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")
print("-" * 30)

# Step 8: Train Ridge Regression
print("\nStep 8: Training Ridge Regression model...")
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train_scaled, y_train)
print("Ridge Regression model trained successfully.")
print("-" * 30)

# Step 9: Train Random Forest
print("\nStep 9: Training Random Forest Regressor model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train) # Using unscaled X_train
print("Random Forest Regressor model trained successfully.")
print("-" * 30)

# Step 10: Train XGBoost
print("\nStep 10: Training XGBoost Regressor model...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, early_stopping_rounds=10)
eval_set = [(X_test, y_test)] # Using X_test as evaluation set for early stopping
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
print("XGBoost Regressor model trained successfully.")
print("-" * 30)

# Step 11: Make predictions on the test set
print("\nStep 11: Making predictions...")
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
print("Predictions made successfully.")
print("-" * 30)

# Step 12: Calculate and print evaluation metrics
print("\nStep 12: Evaluating models...")
models_preds = {
    "Ridge": y_pred_ridge,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}
evaluation_results = {}
for name, y_pred in models_preds.items():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evaluation_results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"\nMetrics for {name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R-squared: {r2:.4f}")
print("-" * 30)

# Step 13: Get and print feature importances
print("\nStep 13: Feature Importances...")

# Random Forest Feature Importances
print("\nRandom Forest - Top 20 Feature Importances:")
if hasattr(rf_model, 'feature_importances_'):
    importances_rf = rf_model.feature_importances_
    rf_feature_importances = pd.Series(importances_rf, index=X_column_names).sort_values(ascending=False)
    print(rf_feature_importances.head(20))
else:
    print("Could not retrieve feature importances for Random Forest.")
print("-" * 10)

# XGBoost Feature Importances
print("\nXGBoost - Top 20 Feature Importances:")
if hasattr(xgb_model, 'feature_importances_'):
    importances_xgb = xgb_model.feature_importances_
    xgb_feature_importances = pd.Series(importances_xgb, index=X_column_names).sort_values(ascending=False)
    print(xgb_feature_importances.head(20))
else:
    print("Could not retrieve feature importances for XGBoost.")
print("-" * 30)

# Step 14: Libraries are imported at the top.

print("Model evaluation script (final features) finished.")
