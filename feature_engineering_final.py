import pandas as pd
import numpy as np

# Step 1: Load wr_data_with_college.parquet
print("Step 1: Loading wr_data_with_college.parquet...")
try:
    df = pd.read_parquet("wr_data_with_college.parquet")
    print("Successfully loaded wr_data_with_college.parquet.")
    print(f"Initial DataFrame shape: {df.shape}")
except FileNotFoundError:
    print("Error: wr_data_with_college.parquet not found. Please ensure this file exists.")
    exit()
except Exception as e:
    print(f"Error loading wr_data_with_college.parquet: {e}")
    exit()
print("-" * 30)

# Step 2: Create college_stats_missing indicator
# This should be based on a column that would only be non-NaN if college stats were successfully merged.
# 'college_data_source' was created in the previous step for this.
print("\nStep 2: Create 'college_stats_missing' indicator...")
if 'college_data_source' in df.columns: # This column indicates a successful merge of college data
    df['college_stats_missing'] = df['college_data_source'].isnull().astype(int)
    print("'college_stats_missing' indicator created based on 'college_data_source'.")
else:
    # Fallback if 'college_data_source' is not there (e.g. if dummy columns were added differently)
    # If 'college_games_played' is 0 and other college stats are 0, it might also mean missing,
    # but the prompt implies it's based on NaN before imputation.
    # The dummy columns were filled with 0, not NaN.
    # For robustness, if 'college_games_played' is one of the new columns, check its initial state.
    # However, the prompt says "if college_games_played is NaN", and we are about to fill it.
    # The most reliable indicator is 'college_data_source' from the previous step.
    # If that's missing, it implies the previous step didn't run as expected.
    # For now, let's assume 'college_data_source' is the definitive source.
    print("Warning: 'college_data_source' column not found. Cannot accurately create 'college_stats_missing'. Assuming all missing for safety.")
    df['college_stats_missing'] = 1
print(df['college_stats_missing'].value_counts())
print("-" * 30)

# Step 3: Impute NaNs in raw college stat columns with 0
print("\nStep 3: Impute NaNs in raw college stat columns with 0...")
college_stat_cols_raw = ['college_games_played', 'college_receptions',
                         'college_receiving_yards', 'college_receiving_tds']
for col in college_stat_cols_raw:
    if col in df.columns:
        df[col] = df[col].fillna(0)
        print(f"Imputed NaNs in '{col}' with 0.")
    else:
        print(f"Warning: College stat column '{col}' not found. Creating it with 0s.")
        df[col] = 0 # Ensure column exists for derived feature calculation
print("-" * 30)

# Step 4: Create derived college features
print("\nStep 4: Create derived college features...")
df['college_yards_per_reception'] = (df['college_receiving_yards'] / df['college_receptions'])
df['college_tds_per_reception'] = (df['college_receiving_tds'] / df['college_receptions'])
df['college_yards_per_game'] = (df['college_receiving_yards'] / df['college_games_played'])

# Handle division by zero (results in inf) and NaNs (if any intermediate NaNs occurred)
for col in ['college_yards_per_reception', 'college_tds_per_reception', 'college_yards_per_game']:
    df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    print(f"Created and cleaned derived college feature: '{col}'.")
print(df[['college_receptions', 'college_receiving_yards', 'college_yards_per_reception']].head())
print("-" * 30)

# Step 5: Drop leaky features
print("\nStep 5: Drop leaky features...")
leaky_cols = ['w_av', 'w_av_missing', 'dr_av', 'dr_av_missing']
cols_to_drop_leaky = [col for col in leaky_cols if col in df.columns]
if cols_to_drop_leaky:
    df = df.drop(columns=cols_to_drop_leaky)
    print(f"Dropped leaky columns: {cols_to_drop_leaky}")
else:
    print("No leaky columns (w_av, dr_av and their missing indicators) found to drop.")
print("-" * 30)

# Step 6: Calculate age_at_draft
print("\nStep 6: Calculate 'age_at_draft'...")
if 'season' in df.columns and 'birth_date' in df.columns:
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    df['age_at_draft_missing'] = df['birth_date'].isnull().astype(int)
    df['age_at_draft'] = df['season'] - df['birth_date'].dt.year

    age_median = df['age_at_draft'].median()
    df['age_at_draft'] = df['age_at_draft'].fillna(age_median).astype(int)
    print(f"Calculated 'age_at_draft', imputed missing with median ({age_median}).")
    print(df[['season', 'age_at_draft', 'age_at_draft_missing']].head())
else:
    print("Could not calculate 'age_at_draft': 'season' or 'birth_date' column missing.")
    df['age_at_draft'] = np.nan # Ensure column exists if it couldn't be calculated
    df['age_at_draft_missing'] = 1
print("-" * 30)

# Step 7: Calculate BMI
print("\nStep 7: Calculate 'bmi'...")
if 'height' in df.columns and 'weight' in df.columns:
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

    df['bmi_missing'] = (df['height'].isnull() | df['weight'].isnull()).astype(int)
    df['bmi'] = np.where(
        (df['height'].notna() & df['weight'].notna() & (df['height'] > 0)),
        (df['weight'] / (df['height']**2)) * 703,
        np.nan
    )
    bmi_median = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(bmi_median)
    print(f"Calculated 'bmi', imputed missing with median ({bmi_median}).")
    print(df[['height', 'weight', 'bmi', 'bmi_missing']].head())
else:
    print("Could not calculate 'bmi': 'height' or 'weight' column missing.")
    df['bmi'] = np.nan # Ensure column exists
    df['bmi_missing'] = 1
print("-" * 30)

# Step 8: Handle Missing Values for Combine Stats
print("\nStep 8: Handling Missing Values for Combine Stats...")
combine_cols = ['forty', 'vertical', 'bench', 'broad_jump', 'cone', 'shuttle', 'wt'] # 'wt' is combine weight
# 'ht' (combine height) was previously noted as all NaN and should be dropped if still present.
if 'ht' in df.columns:
    df = df.drop(columns=['ht'])
    print("Dropped 'ht' (combine height) column as it's expected to be all NaN.")

for col in combine_cols:
    if col in df.columns:
        if df[col].isnull().any():
            indicator_col = f'{col}_missing'
            if indicator_col not in df.columns: # Create if not already present from previous step
                 df[indicator_col] = df[col].isnull().astype(int)

            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Imputed missing values in '{col}' with median ({median_val}). Corresponding '{indicator_col}' used/created.")
        elif f'{col}_missing' not in df.columns: # No missing values, ensure indicator exists
             df[f'{col}_missing'] = 0
    else:
        print(f"Warning: Combine column '{col}' not found.")
print("-" * 30)

# Step 9: Categorical Feature Encoding for 'college' (assuming 'college' is the main one)
print("\nStep 9: Categorical Feature Encoding for 'college'...")
# The main college column from draft_picks was 'college'.
# 'college_name' from player_info was also present.
# The previous feature_engineering script used 'college' and then dropped it.
# If 'college_name' is still present, use it. Otherwise, this step might not find a column.
college_col_to_encode = None
if 'college' in df.columns: # This was dropped in the *previous* feature engineering script.
    college_col_to_encode = 'college'
elif 'college_name' in df.columns: # This might also have been dropped.
    college_col_to_encode = 'college_name'

if college_col_to_encode and college_col_to_encode in df.columns:
    df[college_col_to_encode] = df[college_col_to_encode].fillna('Unknown')
    college_freq = df[college_col_to_encode].value_counts(normalize=True)
    df['college_freq_encoded'] = df[college_col_to_encode].map(college_freq)
    print(f"'{college_col_to_encode}' frequency encoded into 'college_freq_encoded'.")
     # Original column will be dropped in Step 10
else:
    print(f"No primary college column ('college' or 'college_name') found for encoding, or it was already encoded and dropped.")
    if 'college_freq_encoded' not in df.columns: # If it wasn't created in previous script
        df['college_freq_encoded'] = 0 # Create dummy if not found
        print("Created dummy 'college_freq_encoded' as 0.")
print("-" * 30)


# Step 10: Final Feature Selection & Cleanup
print("\nStep 10: Final Feature Selection & Cleanup...")
if 'target_AV' not in df.columns:
    print("CRITICAL ERROR: 'target_AV' is missing before attempting to separate it.")
    exit()
y = df['target_AV'].copy()
X = df.drop(columns=['target_AV'])

# Comprehensive NaN imputation for any remaining numeric columns in X
print("\nComprehensive NaN imputation for X features...")
numeric_cols_in_X = X.select_dtypes(include=np.number).columns
for col in numeric_cols_in_X:
    if X[col].isnull().any():
        missing_indicator_col_name = f"{col}_missing"
        if missing_indicator_col_name not in X.columns: # Check if feature_engineering already created it
             X[missing_indicator_col_name] = X[col].isnull().astype(int)
             print(f"Created '{missing_indicator_col_name}' indicator for column '{col}'.")

        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"Imputed NaNs in '{col}' with median ({median_val}).")
print("Finished comprehensive NaN imputation for X features.")
print(f"Missing values in X after comprehensive imputation: {X.isnull().sum().sum()}")


# Define columns to drop (identifiers, raw text, specific handled columns)
cols_to_drop_final = []
identifiers = ['pfr_player_id', 'gsis_id', 'cfb_player_id', 'esb_id', 'gsis_it_id', 'smart_id', 'player_id']
descriptive_text = ['pfr_player_name', 'display_name', 'first_name', 'last_name', 'short_name', 'football_name',
                    'team', 'category', 'side', 'college_conference', 'current_team_id', 'draft_club',
                    'headshot', 'status', 'status_description_abbr', 'status_short_description',
                    'team_abbr', 'uniform_number', 'suffix', 'college_data_source', 'player_name',
                    'college_team_name']
dates = ['birth_date']
original_college_names = []
if college_col_to_encode and college_col_to_encode in X.columns:
    original_college_names.append(college_col_to_encode)
# Ensure other potential original college name columns are also dropped if they exist
if 'college_name' in X.columns and 'college_name' not in original_college_names : original_college_names.append('college_name')
if 'college' in X.columns and 'college' not in original_college_names : original_college_names.append('college')

position_cols = ['position_draft', 'position_player', 'position', 'pos', 'position_group']
other_to_drop = ['car_av', 'years_of_experience'] # Add car_av (all NaN) and years_of_experience (object type)

cols_to_drop_final.extend(identifiers)
cols_to_drop_final.extend(descriptive_text)
cols_to_drop_final.extend(dates)
cols_to_drop_final.extend(original_college_names)
cols_to_drop_final.extend(position_cols)
cols_to_drop_final.extend(other_to_drop)

final_cols_to_drop_existing = list(set(col for col in cols_to_drop_final if col in X.columns))
if final_cols_to_drop_existing:
    X = X.drop(columns=final_cols_to_drop_existing)
    print(f"Dropped columns: {final_cols_to_drop_existing}")
else:
    print("No columns from the explicit drop list were found in X.")

print(f"X shape after final cleanup: {X.shape}")
print("-" * 30)

# Step 11: Print .info(), .head(), and missing value counts
print("\nStep 11: Final DataFrame Info & Missing Values...")
print("X (features) .info():")
X.info(verbose=True, show_counts=True)
print("\nX (features) missing value summary:")
print(X.isnull().sum()[X.isnull().sum() > 0])
print("\nX (features) .head():")
print(X.head())

print("\ny (target) .info():")
y.info()
print("\ny (target) missing value summary:")
print(y.isnull().sum())
print("\ny (target) .head():")
print(y.head())
print("-" * 30)

# Step 12: Concatenate X and y and save as wr_features_final.parquet
print("\nStep 12: Saving final feature set...")
final_df_to_save = pd.concat([X, y], axis=1)
try:
    final_df_to_save.to_parquet("wr_features_final.parquet", index=False)
    print("DataFrame saved successfully to wr_features_final.parquet.")
    print(f"Final saved shape: {final_df_to_save.shape}")
except Exception as e:
    print(f"Error saving DataFrame to parquet: {e}")
print("-" * 30)

print("Final feature engineering script finished.")
