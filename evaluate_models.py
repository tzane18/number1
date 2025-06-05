import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load wr_features.parquet
print("Step 1: Loading wr_features.parquet...")
try:
    df = pd.read_parquet("wr_features.parquet")
    print("Successfully loaded wr_features.parquet.")
    print(f"DataFrame shape: {df.shape}")
except FileNotFoundError:
    print("Error: wr_features.parquet not found. Please ensure this file exists from the previous subtask.")
    exit()
except Exception as e:
    print(f"Error loading wr_features.parquet: {e}")
    exit()
print("-" * 30)

# Step 2: Separate features (X) and target (y)
print("\nStep 2: Separating features (X) and target (y)...")
if 'target_AV' not in df.columns:
    print("Error: 'target_AV' column not found in the DataFrame.")
    exit()
X = df.drop(columns=['target_AV'])
y = df['target_AV']
print(f"Initial X shape: {X.shape}, y shape: {y.shape}")
print("-" * 30)

# Step 3: Preprocess X
print("\nStep 3: Preprocessing X features...")
# Drop 'ht' column if it exists
if 'ht' in X.columns:
    X = X.drop(columns=['ht'])
    print("Dropped 'ht' column from X features.")

# Convert 'hof' column to int if it exists and is boolean
if 'hof' in X.columns and X['hof'].dtype == 'bool':
    X['hof'] = X['hof'].astype(int)
    print("'hof' column converted to int.")

# Impute remaining NaNs and create _missing indicators
# This replicates the logic from the successful part of train_models.py
# In a pipeline, this would be part of the preprocessing steps.
numeric_cols_in_X = X.select_dtypes(include=np.number).columns
print(f"Numeric columns to check for imputation: {numeric_cols_in_X.tolist()}")

for col in numeric_cols_in_X:
    if X[col].isnull().any():
        missing_indicator_col_name = f"{col}_missing"
        # Add indicator if it wasn't added in feature_engineering.py or already exists from that step
        # The feature_engineering script added indicators for many columns.
        # This ensures any numeric column *still* having NaNs gets an indicator if it doesn't have one.
        if missing_indicator_col_name not in X.columns:
             X[missing_indicator_col_name] = X[col].isnull().astype(int)
             print(f"Created '{missing_indicator_col_name}' indicator for column '{col}'.")

        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"Imputed NaNs in '{col}' with median ({median_val}).")

print(f"Missing values in X after preprocessing: {X.isnull().sum().sum()}")
print(f"X shape after preprocessing: {X.shape}")
print("-" * 30)

# Step 4: Store original X column names
# This should be done *after* all column additions (_missing) and drops (ht)
# but *before* filtering rows based on y.
X_column_names = X.columns.tolist()
print(f"\nStep 4: Stored X column names (count: {len(X_column_names)}). Example: {X_column_names[:5]}")
print("-" * 30)

# Step 5: Filter out rows where target_AV is NaN from both X and y
print("\nStep 5: Filtering rows with NaN target_AV...")
print(f"Number of rows before dropping NaNs in target_AV: {X.shape[0]}")
not_na_target_indices = y.notna()
X = X[not_na_target_indices].copy()
y = y[not_na_target_indices].copy()
print(f"Number of rows after dropping NaNs in target_AV: {X.shape[0]}")
print(f"X shape after filtering for y: {X.shape}, y shape: {y.shape}")
if len(y) == 0:
    print("Error: No data remaining after dropping NaNs from target_AV. Cannot proceed.")
    exit()
print("-" * 30)

# Step 6: Split data into training (80%) and testing (20%) sets
print("\nStep 6: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("-" * 30)

# Step 7: Initialize and fit StandardScaler on X_train. Transform X_train and X_test.
print("\nStep 7: Scaling features...")
scaler = StandardScaler()
# Ensure all columns in X_train/X_test are numeric before scaling
# This should be true due to preprocessing, but as a safeguard:
X_train_numeric = X_train.select_dtypes(include=np.number)
X_test_numeric = X_test.select_dtypes(include=np.number)
# The column names list should align with these numeric versions if all preprocessing was correct
if X_train_numeric.shape[1] != len(X_column_names) or X_test_numeric.shape[1] != len(X_column_names):
    print("Warning: Discrepancy in column counts after numeric selection for scaling vs. stored column names.")
    print(f"Original X_column_names count: {len(X_column_names)}")
    print(f"X_train_numeric columns: {X_train_numeric.shape[1]}, X_test_numeric columns: {X_test_numeric.shape[1]}")
    # Update X_column_names if columns were indeed dropped by select_dtypes (e.g. if an unexpected object col remained)
    if X_train_numeric.shape[1] < len(X_column_names):
         print("Updating X_column_names to match numeric columns being scaled.")
         X_column_names = X_train_numeric.columns.tolist()


X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)
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
# Using X_train (unscaled, but fully imputed numeric columns)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_numeric, y_train) # Use X_train_numeric to match scaled data columns
print("Random Forest Regressor model trained successfully.")
print("-" * 30)

# Step 10: Train XGBoost
print("\nStep 10: Training XGBoost Regressor model...")
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_numeric, y_train) # Use X_train_numeric
print("XGBoost Regressor model trained successfully.")
print("-" * 30)

# Step 11: Make predictions on the test set
print("\nStep 11: Making predictions...")
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test_numeric) # Use X_test_numeric
y_pred_xgb = xgb_model.predict(X_test_numeric) # Use X_test_numeric
print("Predictions made successfully.")
print("-" * 30)

# Step 12: Calculate and print evaluation metrics
print("\nStep 12: Evaluating models...")
models = {
    "Ridge": y_pred_ridge,
    "Random Forest": y_pred_rf,
    "XGBoost": y_pred_xgb
}
for name, y_pred in models.items():
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
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
importances_rf = rf_model.feature_importances_
rf_feature_importances = pd.Series(importances_rf, index=X_column_names).sort_values(ascending=False)
print(rf_feature_importances.head(20))
print("-" * 10)

# XGBoost Feature Importances
print("\nXGBoost - Top 20 Feature Importances:")
importances_xgb = xgb_model.feature_importances_
xgb_feature_importances = pd.Series(importances_xgb, index=X_column_names).sort_values(ascending=False)
print(xgb_feature_importances.head(20))
print("-" * 30)

# Step 14: Libraries are imported at the top.

print("Model evaluation script finished.")
