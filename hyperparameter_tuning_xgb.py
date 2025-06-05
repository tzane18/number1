import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import time

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

# Step 2: Separate features (X) from target (y)
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
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Verify X is purely numeric and no NaNs (should be guaranteed by previous step)
if X.isnull().sum().sum() > 0:
    print(f"WARNING: NaNs found in X after loading! Count: {X.isnull().sum().sum()}. This should have been handled.")
    # Fallback if needed, though previous script should ensure cleanliness
    numeric_cols = X.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            print(f"Emergency imputing NaNs in {col} with median.")
            X[col].fillna(X[col].median(), inplace=True)
    if X.isnull().sum().sum() > 0:
        print("ERROR: NaNs still present in X after emergency fill. Exiting.")
        exit()
print("-" * 30)

# Step 3: Print number of rows before dropping missing target_AV
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

# Step 5: Split data
print("\nStep 5: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("-" * 30)

# Step 5 (from prompt, for param_dist): Define the parameter distribution for RandomizedSearchCV
print("\nStep 6 (from prompt): Defining parameter distribution for XGBoost...")
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.001, 0.01, 0.1],  # L1 regularization
    'reg_lambda': [1, 1.1, 1.2, 1.3]   # L2 regularization (XGBoost default is 1)
}
print("Parameter distribution defined.")
print("-" * 30)

# Step 6 (from prompt): Initialize XGBRegressor
print("\nStep 7 (from prompt): Initializing base XGBRegressor for RandomizedSearchCV...")
# Using tree_method='hist' for potentially faster training, especially on larger datasets
# It's also robust to missing values if any were to slip through (though X should be clean)
base_xgb_model = xgb.XGBRegressor(random_state=42, tree_method='hist')
print(f"Base XGBoost model: {base_xgb_model}")
print("-" * 30)

# Step 7 (from prompt): Initialize RandomizedSearchCV
print("\nStep 8 (from prompt): Initializing RandomizedSearchCV...")
# n_iter=50 means 50 random combinations of parameters will be tried.
# cv=3 means 3-fold cross-validation for each combination.
random_search = RandomizedSearchCV(
    estimator=base_xgb_model,
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=3,       # Number of cross-validation folds
    scoring='r2', # Score to optimize for
    n_jobs=-1,  # Use all available cores
    random_state=42,
    verbose=1   # Shows progress
)
print(f"RandomizedSearchCV initialized: {random_search}")
print("-" * 30)

# Step 8 (from prompt): Fit RandomizedSearchCV
print("\nStep 9 (from prompt): Fitting RandomizedSearchCV...")
start_time = time.time()
random_search.fit(X_train, y_train) # X_train is unscaled, which is fine for XGBoost
end_time = time.time()
print(f"RandomizedSearchCV fitting completed in {end_time - start_time:.2f} seconds.")
print("-" * 30)

# Step 9 (from prompt): Print best parameters
print("\nStep 10 (from prompt): Best parameters found by RandomizedSearchCV:")
best_params = random_search.best_params_
print(best_params)
print("-" * 30)

# Step 10 (from prompt): Print best R2 score from CV
print("\nStep 11 (from prompt): Best R2 score from Cross-Validation:")
best_cv_score = random_search.best_score_
print(f"Best R2 (CV): {best_cv_score:.4f}")
print("-" * 30)

# Step 11 (from prompt): Initialize a new XGBRegressor with best_params_
print("\nStep 12 (from prompt): Initializing and training best XGBoost model with early stopping...")
best_xgb_model = xgb.XGBRegressor(
    **best_params,
    random_state=42,
    early_stopping_rounds=10, # Add early stopping for this final model training
    n_jobs=-1
)

# Step 12 (from prompt): Create eval_set
eval_set = [(X_test, y_test)]

# Step 13 (from prompt): Fit this best model
best_xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
print("Best XGBoost model trained successfully with best parameters and early stopping.")
if hasattr(best_xgb_model, 'best_iteration') and best_xgb_model.best_iteration is not None:
    print(f"Best iteration for tuned XGBoost: {best_xgb_model.best_iteration}")
else:
     print(f"Tuned XGBoost used all {best_xgb_model.get_params()['n_estimators']} estimators (early stopping not triggered or n_estimators from tuning was optimal).")
print("-" * 30)

# Step 14 (from prompt): Make predictions
print("\nStep 13 (from prompt): Making predictions with tuned XGBoost model...")
y_pred_tuned_xgb = best_xgb_model.predict(X_test)
print("Predictions made with tuned XGBoost model.")
print("-" * 30)

# Step 15 (from prompt): Calculate and print evaluation metrics for the tuned XGBoost model
print("\nStep 14 (from prompt): Evaluating tuned XGBoost model...")
mse_tuned_xgb = mean_squared_error(y_test, y_pred_tuned_xgb)
rmse_tuned_xgb = np.sqrt(mse_tuned_xgb)
mae_tuned_xgb = mean_absolute_error(y_test, y_pred_tuned_xgb)
r2_tuned_xgb = r2_score(y_test, y_pred_tuned_xgb)

print("Metrics for Tuned XGBoost Regressor:")
print(f"  MSE: {mse_tuned_xgb:.4f}")
print(f"  RMSE: {rmse_tuned_xgb:.4f}")
print(f"  MAE: {mae_tuned_xgb:.4f}")
print(f"  R-squared: {r2_tuned_xgb:.4f}")
print("-" * 30)

# Step 16 (from prompt): Compare tuned XGBoost R2 with the previous best R2 (Ridge: ~0.486)
print("\nStep 15 (from prompt): Performance Comparison...")
previous_best_r2_ridge = 0.4864 # From previous evaluation subtask
print(f"Previous best R2 (Ridge): {previous_best_r2_ridge:.4f}")
print(f"Tuned XGBoost R2: {r2_tuned_xgb:.4f}")
if r2_tuned_xgb > previous_best_r2_ridge:
    print("Tuned XGBoost model shows improvement over the previous best Ridge model.")
else:
    print("Tuned XGBoost model does not show improvement over the previous best Ridge model based on R2.")
print("-" * 30)

print("Hyperparameter tuning script finished.")
