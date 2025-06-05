import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb # Needs to be imported as xgb for XGBRegressor

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

# Step 2: Separate features (X) from the target variable target_AV (y)
print("\nStep 2: Separating features (X) and target (y)...")
if 'target_AV' not in df.columns:
    print("Error: 'target_AV' column not found in the DataFrame.")
    exit()

X = df.drop(columns=['target_AV'])
y = df['target_AV']

# Explicitly drop 'ht' column if it exists, as it's all NaN and causing issues
if 'ht' in X.columns:
    X = X.drop(columns=['ht'])
    print("Dropped 'ht' column from X features as it's expected to be all NaN.")

print(f"X shape after initial processing: {X.shape}, y shape: {y.shape}")

# Convert 'hof' (boolean) to int if it exists
if 'hof' in X.columns:
    X['hof'] = X['hof'].astype(int)
    print("'hof' column converted to int.")

# Pre-imputation of remaining NaNs in X before splitting and scaling
print("\nPre-imputing remaining NaNs in X features...")
numeric_cols_in_X = X.select_dtypes(include=np.number).columns
for col in numeric_cols_in_X:
    if X[col].isnull().any():
        # Create _missing indicator only if one wasn't likely created in feature_engineering.py
        # (i.e., if col itself doesn't end with _missing and no col_missing exists)
        missing_indicator_col_name = f"{col}_missing"
        if not col.endswith('_missing') and missing_indicator_col_name not in X.columns:
            X[missing_indicator_col_name] = X[col].isnull().astype(int)
            print(f"Created '{missing_indicator_col_name}' indicator for column '{col}'.")

        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"Imputed NaNs in '{col}' with median ({median_val}).")
print("Finished pre-imputation of X features.")
print(f"Missing values in X after pre-imputation: {X.isnull().sum().sum()}")

print("-" * 30)

# Step 3: Print the number of rows before dropping missing target_AV
print("\nStep 3: Handling missing target_AV values...")
print(f"Number of rows before dropping NaNs in target_AV: {len(df)}") # df still includes original y

# Step 4: Remove rows where target_AV is NaN
# Also, ensure X and y are aligned after this drop
not_na_target_indices = y.notna()
X = X[not_na_target_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
y = y[not_na_target_indices].copy()
print(f"Number of rows after dropping NaNs in target_AV: {len(y)}")
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
# X_train and X_test should be purely numeric at this point due to earlier select_dtypes and imputation.
# However, ensure all columns are numeric before scaling, just in case.
X_train_numeric = X_train.select_dtypes(include=np.number)
X_test_numeric = X_test.select_dtypes(include=np.number)

if X_train_numeric.shape[1] != X_train.shape[1]:
    print(f"Warning: Some non-numeric columns were dropped from X_train just before scaling. Original X_train had {X_train.shape[1]} features, X_train_numeric now has {X_train_numeric.shape[1]}.")
    dropped_cols_pre_scale = set(X_train.columns) - set(X_train_numeric.columns)
    print(f"Columns dropped pre-scaling: {dropped_cols_pre_scale}")
else:
    print("All columns in X_train are numeric, proceeding with scaling.")


print(f"Fitting StandardScaler on X_train_numeric (shape: {X_train_numeric.shape})...")
X_train_scaled = scaler.fit_transform(X_train_numeric)
print("Transforming X_test_numeric...")
X_test_scaled = scaler.transform(X_test_numeric)
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
# Random Forest does not strictly require scaled features, so using original X_train
try:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train) # Using X_train (unscaled, but numeric only)
    print("Random Forest Regressor model trained successfully.")
except Exception as e:
    print(f"Error training Random Forest Regressor model: {e}")
print("-" * 30)

# Step 10: Train an XGBoost Regressor model
print("\nStep 10: Training XGBoost Regressor model...")
# XGBoost also does not strictly require scaled features
try:
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train) # Using X_train (unscaled, but numeric only)
    print("XGBoost Regressor model trained successfully.")
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

print("\nModel training script finished.")
