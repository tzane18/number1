import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# Step 2: Separate features (X) from target (y). Store original X column names.
print("\nStep 2: Separating features (X) and target (y)...")
if 'target_AV' not in df.columns:
    print("Error: 'target_AV' column not found in the DataFrame.")
    exit()

X = df.drop(columns=['target_AV'])
y = df['target_AV']

# Ensure 'hof' is int, as it might be boolean from parquet
if 'hof' in X.columns and X['hof'].dtype == 'bool':
    X['hof'] = X['hof'].astype(int)
    print("'hof' column converted to int type for consistency.")

X_column_names = X.columns.tolist() # Store after 'hof' conversion
print(f"X shape: {X.shape}, y shape: {y.shape}. Stored {len(X_column_names)} feature names.")

# Verify no NaNs in X (should be guaranteed by feature_engineering_predraft.py)
if X.isnull().sum().sum() > 0:
    print(f"WARNING: NaNs found in X after loading! Count: {X.isnull().sum().sum()}. This should have been handled.")
    # Fallback imputation if necessary (should not be needed if previous step was perfect)
    numeric_cols = X.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            print(f"Emergency imputing NaNs in {col} with median for this run.")
            X[col].fillna(X[col].median(), inplace=True)
    if X.isnull().sum().sum() > 0:
        print("ERROR: NaNs still present in X after emergency fill. Exiting.")
        exit()
print("-" * 30)

# Step 3: Print number of rows before dropping missing target_AV
print("\nStep 3: Handling missing target_AV values...")
print(f"Number of rows before dropping NaNs in target_AV: {X.shape[0]}")

# Step 4: Remove rows where target_AV is NaN from X and y
not_na_target_indices = y.notna()
X = X[not_na_target_indices].copy()
y = y[not_na_target_indices].copy()
print(f"Number of rows after dropping NaNs in target_AV: {X.shape[0]}")
print(f"X shape after filtering for y: {X.shape}, y shape: {y.shape}")
if len(y) == 0:
    print("Error: No data remaining after dropping NaNs from target_AV. Cannot proceed.")
    exit()
print("-" * 30)

# Step 5: Split data
print("\nStep 5: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
# Verify column name list length
if len(X_column_names) != X_train.shape[1]:
    print(f"Warning: Column name list length ({len(X_column_names)}) does not match X_train feature count ({X_train.shape[1]}). Resetting column names from X_train.")
    X_column_names = X_train.columns.tolist() # This ensures feature importances match the data used by RF/XGB
print("-" * 30)

# Step 6 & 7: Initialize and fit StandardScaler, transform data
print("\nStep 6 & 7: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")
print(f"X_train_scaled shape: {X_train_scaled.shape}, X_test_scaled shape: {X_test_scaled.shape}")
print("-" * 30)

# Step 6 (repeated for clarity of prompt): Train Ridge Regression (as per prompt steps)
print("\nStep 8 (Re-numbered from prompt's 6): Training Ridge Regression model...")
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train_scaled, y_train)
print("Ridge Regression model trained successfully.")
print("-" * 30)

# Step 7 (repeated): Train Random Forest
print("\nStep 9 (Re-numbered from prompt's 7): Training Random Forest Regressor model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train) # Using unscaled X_train
print("Random Forest Regressor model trained successfully.")
print("-" * 30)

# Step 8 (repeated): Train XGBoost
print("\nStep 10 (Re-numbered from prompt's 8): Training XGBoost Regressor model...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, early_stopping_rounds=10)
eval_set = [(X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
print("XGBoost Regressor model trained successfully.")
if xgb_model.best_iteration is not None: # Check if early stopping occurred
    print(f"Best iteration for XGBoost: {xgb_model.best_iteration}")
else: # n_estimators was reached
    print(f"XGBoost used all {xgb_model.get_params()['n_estimators']} estimators (early stopping not triggered or n_estimators too low).")

print("-" * 30)

# Step 9 (prompt): Make predictions
print("\nStep 11 (Re-numbered from prompt's 9): Making predictions...")
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
print("Predictions made successfully.")
print("-" * 30)

# Step 10 (prompt): Calculate and print evaluation metrics
print("\nStep 12 (Re-numbered from prompt's 10): Evaluating models...")
models_preds = {
    "Ridge": y_pred_ridge,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}
evaluation_results = {}
best_model_name = ""
best_r2 = -float('inf')

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
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
print("-" * 30)

# Step 11 (prompt): Get and print feature importances
print("\nStep 13 (Re-numbered from prompt's 11): Feature Importances...")
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

# Step 12 (prompt): Summary statement
print("\nStep 14 (Re-numbered from prompt's 12): Summary Statement...")
print(f"The model with the best R-squared score on the test set is: {best_model_name} (R2 = {best_r2:.4f}).")
if best_r2 > 0.3:
    print("An R-squared value in this range suggests a moderate to good level of predictability of target_AV using pre-draft features.")
elif best_r2 > 0.1:
    print("An R-squared value in this range suggests a low but potentially non-negligible level of predictability.")
else:
    print("An R-squared value in this range suggests very low predictability of target_AV using the current pre-draft features.")
print("Further feature engineering, hyperparameter tuning, or trying different model types might improve performance.")
print("-" * 30)

print("Model evaluation script (pre-draft features) finished.")
