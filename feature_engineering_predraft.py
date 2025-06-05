import pandas as pd
import numpy as np

# Step 1: Load wr_data_with_college.parquet
print("Step 1: Loading wr_data_with_college.parquet...")
try:
    df_full = pd.read_parquet("wr_data_with_college.parquet")
    print("Successfully loaded wr_data_with_college.parquet.")
    print(f"Initial DataFrame shape: {df_full.shape}")
except FileNotFoundError:
    print("Error: wr_data_with_college.parquet not found. Please ensure this file exists.")
    exit()
except Exception as e:
    print(f"Error loading wr_data_with_college.parquet: {e}")
    exit()
print("-" * 30)

# Step 2: Separate target_AV and create X_features
print("\nStep 2: Separating target and creating X_features copy...")
if 'target_AV' not in df_full.columns:
    print("Error: 'target_AV' column not found.")
    exit()
y_true_target = df_full['target_AV'].copy()
X_features = df_full.drop(columns=['target_AV'])
print(f"X_features shape: {X_features.shape}, y_true_target shape: {y_true_target.shape}")
print("-" * 30)

# Step 3: Identify and DROP NFL Career Summary Stats from X_features
print("\nStep 3: Dropping NFL Career Summary Stats...")
# Columns from draft_picks that are post-draft career summaries
# Also include their _missing indicators if they were created in previous feature engineering steps
nfl_career_stats_cols = [
    'receptions', 'rec_yards', 'rec_tds',
    'games', 'w_av', 'dr_av', 'car_av',
    'pass_completions', 'pass_attempts', 'pass_yards', 'pass_tds', 'pass_ints', # Using pass_interceptions from prompt
    'rush_atts', 'rush_yards', 'rush_tds', # Using rush_attempts from prompt
    # 'position', # This is usually pre-draft position, but let's list it. If it's NFL pos, it's leaky.
                # The feature_engineering_final script already handled dropping position_draft if all 'WR'.
    # 'award_summary', # This column wasn't in nfl_data_py's draft_picks output.
    'to', # 'to' year of draft_picks data often indicates end of recorded stats for that player in file
    'allpro', 'probowls', 'seasons_started',
    # Missing indicators for these if they exist
    'receptions_missing', 'rec_yards_missing', 'rec_tds_missing',
    'games_missing', 'w_av_missing', 'dr_av_missing', 'car_av_missing',
    'pass_completions_missing', 'pass_attempts_missing', 'pass_yards_missing',
    'pass_tds_missing', 'pass_ints_missing',
    'rush_atts_missing', 'rush_yards_missing', 'rush_tds_missing',
    'to_missing', 'allpro_missing', 'probowls_missing', 'seasons_started_missing',
    # From previous feature engineering, some of these might have been imputed then dropped
    # This list aims to be comprehensive for raw career stats from draft_picks.
]
# Add `pass_interceptions` and `rush_attempts` if they exist, as per prompt
if 'pass_interceptions' in X_features.columns: nfl_career_stats_cols.append('pass_interceptions')
if 'pass_interceptions_missing' in X_features.columns: nfl_career_stats_cols.append('pass_interceptions_missing')
if 'rush_attempts' in X_features.columns: nfl_career_stats_cols.append('rush_attempts')
if 'rush_attempts_missing' in X_features.columns: nfl_career_stats_cols.append('rush_attempts_missing')


cols_actually_dropped_career = [col for col in nfl_career_stats_cols if col in X_features.columns]
if cols_actually_dropped_career:
    X_features = X_features.drop(columns=cols_actually_dropped_career)
    print(f"Dropped NFL career summary stat columns: {cols_actually_dropped_career}")
else:
    print("No specified NFL career summary stat columns found to drop.")
print(f"X_features shape after dropping career stats: {X_features.shape}")
print("-" * 30)

# Step 4: Create college_stats_missing indicator
print("\nStep 4: Create 'college_stats_missing' indicator...")
if 'college_data_source' in X_features.columns:
    X_features['college_stats_missing'] = X_features['college_data_source'].isnull().astype(int)
    print("'college_stats_missing' created from 'college_data_source'.")
elif 'college_games_played' in X_features.columns: # Fallback if source col is missing
    # This assumes college_games_played would be NaN if data wasn't merged/found,
    # but previous step initialized these with 0.
    # A better check might be if college_games_played is 0 AND other key stats are 0.
    # For now, if college_data_source is gone, assume this means no college data was processed.
    # The prompt "if college_games_played is NaN" implies checking before imputation.
    # If this script is run on data from previous subtask (download_merge), college_games_played is 0 not NaN for missing.
    # Thus, this indicator will be mostly 0 if college_data_source isn't there.
    # The safest is to rely on college_data_source. If it's not there, this step might be inaccurate.
    print("Warning: 'college_data_source' not found. Creating 'college_stats_missing' based on 'college_games_played' being 0 (less reliable).")
    X_features['college_stats_missing'] = (X_features['college_games_played'] == 0).astype(int)
else:
    print("Warning: No basis for 'college_stats_missing'. Setting to 1 (all missing).")
    X_features['college_stats_missing'] = 1
print(X_features['college_stats_missing'].value_counts())
print("-" * 30)

# Step 5: Impute NaNs in raw college stat columns with 0
print("\nStep 5: Impute NaNs in raw college stat columns with 0...")
college_stat_cols_raw = ['college_games_played', 'college_receptions',
                         'college_receiving_yards', 'college_receiving_tds']
for col in college_stat_cols_raw:
    if col in X_features.columns:
        X_features[col] = X_features[col].fillna(0) # Should already be 0 from previous step if merge happened
        print(f"Ensured NaNs in '{col}' are 0.")
    else:
        print(f"Warning: College stat column '{col}' not found for imputation. Creating with 0s.")
        X_features[col] = 0
print("-" * 30)

# Step 6: Create derived college features
print("\nStep 6: Create derived college features...")
X_features['college_yards_per_reception'] = (X_features['college_receiving_yards'] / X_features['college_receptions'])
X_features['college_tds_per_reception'] = (X_features['college_receiving_tds'] / X_features['college_receptions'])
X_features['college_yards_per_game'] = (X_features['college_receiving_yards'] / X_features['college_games_played'])
for col in ['college_yards_per_reception', 'college_tds_per_reception', 'college_yards_per_game']:
    X_features[col] = X_features[col].replace([np.inf, -np.inf], 0).fillna(0)
    print(f"Created and cleaned derived college feature: '{col}'.")
print("-" * 30)

# Step 7: Calculate age_at_draft
print("\nStep 7: Calculate 'age_at_draft'...")
if 'season' in X_features.columns and 'birth_date' in X_features.columns:
    X_features['birth_date_dt'] = pd.to_datetime(X_features['birth_date'], errors='coerce')
    X_features['age_at_draft_missing'] = X_features['birth_date_dt'].isnull().astype(int)
    X_features['age_at_draft'] = X_features['season'] - X_features['birth_date_dt'].dt.year
    age_median = X_features['age_at_draft'].median()
    X_features['age_at_draft'] = X_features['age_at_draft'].fillna(age_median).astype(float)
    print(f"Calculated 'age_at_draft', imputed with median ({age_median}).")
else:
    print("Could not calculate 'age_at_draft': 'season' or 'birth_date' column missing.")
    X_features['age_at_draft'] = X_features.get('age_at_draft', pd.Series(np.nan, index=X_features.index)) # Ensure column exists
    if 'age_at_draft_missing' not in X_features.columns: X_features['age_at_draft_missing'] = 1
print("-" * 30)

# Step 8: Calculate BMI
print("\nStep 8: Calculate 'bmi'...")
if 'height' in X_features.columns and 'weight' in X_features.columns:
    X_features['height'] = pd.to_numeric(X_features['height'], errors='coerce')
    X_features['weight'] = pd.to_numeric(X_features['weight'], errors='coerce')
    X_features['bmi_missing'] = (X_features['height'].isnull() | X_features['weight'].isnull()).astype(int)
    X_features['bmi'] = np.where(
        (X_features['height'].notna() & X_features['weight'].notna() & (X_features['height'] > 0)),
        (X_features['weight'] / (X_features['height']**2)) * 703, np.nan
    )
    bmi_median = X_features['bmi'].median()
    X_features['bmi'] = X_features['bmi'].fillna(bmi_median)
    print(f"Calculated 'bmi', imputed with median ({bmi_median}).")
else:
    print("Could not calculate 'bmi': 'height' or 'weight' column missing.")
    X_features['bmi'] = X_features.get('bmi', pd.Series(np.nan, index=X_features.index))
    if 'bmi_missing' not in X_features.columns: X_features['bmi_missing'] = 1
print("-" * 30)

# Step 9: Handle Missing Values for Combine Stats
print("\nStep 9: Handling Missing Values for Combine Stats...")
combine_cols = ['forty', 'vertical', 'bench', 'broad_jump', 'cone', 'shuttle', 'wt']
if 'ht' in X_features.columns: # ht from combine is usually string, not used if 'height' (numeric) is primary
    X_features = X_features.drop(columns=['ht'], errors='ignore')
    print("Dropped 'ht' (combine height string) column if it existed.")

for col in combine_cols:
    if col in X_features.columns:
        indicator_col = f'{col}_missing'
        if indicator_col not in X_features.columns:
             X_features[indicator_col] = X_features[col].isnull().astype(int)
        median_val = X_features[col].median() # Calculate median on potentially already partially imputed data
        X_features[col].fillna(median_val, inplace=True)
        print(f"Imputed NaNs in '{col}' with median ({median_val}). Indicator '{indicator_col}' ensured.")
    else:
        print(f"Warning: Combine column '{col}' not found.")
print("-" * 30)

# Step 10: Categorical Feature Encoding
print("\nStep 10: Categorical Feature Encoding for original college name...")
# Assuming 'college' from draft_picks is the target for encoding, if it survived previous drops
# Or 'college_name' from player_info.
# The previous script (feature_engineering_final) already did this for 'college' and saved it as 'college_freq_encoded'.
# If that column is present, we can use it. Otherwise, re-create.
college_name_original_col = None
if 'college' in X_features.columns: # This was the one used in previous script.
    college_name_original_col = 'college'
elif 'college_name' in X_features.columns:
    college_name_original_col = 'college_name'

if 'college_freq_encoded' in X_features.columns:
    print("'college_freq_encoded' already exists. Skipping re-creation.")
elif college_name_original_col and college_name_original_col in X_features.columns:
    X_features[college_name_original_col] = X_features[college_name_original_col].fillna('Unknown')
    college_freq = X_features[college_name_original_col].value_counts(normalize=True)
    X_features['college_freq_encoded'] = X_features[college_name_original_col].map(college_freq)
    print(f"Encoded '{college_name_original_col}' into 'college_freq_encoded'.")
else:
    print("No suitable original college column found for encoding or already encoded. Creating dummy 'college_freq_encoded'.")
    X_features['college_freq_encoded'] = 0.0 # Fallback
print("-" * 30)

# Step 11: Final Feature Selection & Cleanup for X_features
print("\nStep 11: Final column cleanup for X_features...")
cols_to_drop_final_predraft = []
# Identifiers (some might have been dropped already)
identifiers = ['pfr_player_id', 'gsis_id', 'cfb_player_id', 'esb_id', 'gsis_it_id', 'smart_id',
               'player_id', 'athlete_id'] # player_id from college data processing
descriptive_text = ['pfr_player_name', 'display_name', 'first_name', 'last_name', 'short_name', 'football_name',
                    'team', 'category', 'side', 'college_conference', 'current_team_id', 'draft_club',
                    'headshot', 'status', 'status_description_abbr', 'status_short_description',
                    'team_abbr', 'uniform_number', 'suffix', 'college_data_source', 'player_name',
                    'college_team_name', 'home_city', 'home_state', 'home_country', 'home_latitude',
                    'home_longitude', 'home_county_fips', 'recruit_ids', 'headshot_url']
dates = ['birth_date', 'birth_date_dt'] # birth_date_dt was temporary
original_college_names_to_drop = []
if college_name_original_col and college_name_original_col in X_features.columns:
    original_college_names_to_drop.append(college_name_original_col)
# Add other variants if they exist and weren't the one encoded
if 'college' in X_features.columns and 'college' not in original_college_names_to_drop: original_college_names_to_drop.append('college')
if 'college_name' in X_features.columns and 'college_name' not in original_college_names_to_drop: original_college_names_to_drop.append('college_name')

position_cols = ['position_draft', 'position_player', 'position', 'pos', 'position_group'] # 'position' from roster too
other_to_drop = ['years_of_experience', 'rookie_year', 'entry_year', 'team_seq'] # NFL specific post-draft or less relevant

cols_to_drop_final_predraft.extend(identifiers)
cols_to_drop_final_predraft.extend(descriptive_text)
cols_to_drop_final_predraft.extend(dates)
cols_to_drop_final_predraft.extend(original_college_names_to_drop)
cols_to_drop_final_predraft.extend(position_cols)
cols_to_drop_final_predraft.extend(other_to_drop)

final_cols_to_drop_predraft_existing = list(set(col for col in cols_to_drop_final_predraft if col in X_features.columns))
if final_cols_to_drop_predraft_existing:
    X_features = X_features.drop(columns=final_cols_to_drop_predraft_existing)
    print(f"Dropped columns for pre-draft feature set: {final_cols_to_drop_predraft_existing}")
else:
    print("No specified columns found to drop for pre-draft feature set.")
print(f"X_features shape after pre-draft cleanup: {X_features.shape}")
print("-" * 30)

# Step 12: Ensure X_features has no remaining NaNs
print("\nStep 12: Final check and imputation for NaNs in X_features...")
numeric_cols_in_X_final = X_features.select_dtypes(include=np.number).columns
final_nan_counts = X_features[numeric_cols_in_X_final].isnull().sum()
cols_with_final_nans = final_nan_counts[final_nan_counts > 0]

if not cols_with_final_nans.empty:
    print(f"Found columns with NaNs after cleanup, applying final median imputation: {cols_with_final_nans.index.tolist()}")
    for col in cols_with_final_nans.index:
        indicator_col = f'{col}_missing' # Ensure indicators exist if created this late
        if indicator_col not in X_features.columns:
             X_features[indicator_col] = X_features[col].isnull().astype(int)
        median_val = X_features[col].median()
        X_features[col].fillna(median_val, inplace=True)
        print(f"Final imputation for '{col}' with median ({median_val}).")
else:
    print("No NaNs found in numeric columns of X_features after cleanup.")
# Check for any non-numeric columns left, besides known objects if any
obj_cols = X_features.select_dtypes(include=['object']).columns
if len(obj_cols) > 0:
    print(f"Warning: Object type columns still present in X_features: {obj_cols.tolist()}. These should be handled or dropped.")
print(f"Final check, missing values in X_features: {X_features.isnull().sum().sum()}")
print("-" * 30)


# Step 13: Print .info() for X_features
print("\nStep 13: Final X_features .info() and columns...")
X_features.info(verbose=True, show_counts=True)
print("\nFinal columns in X_features:")
print(X_features.columns.tolist())
print("-" * 30)

# Step 14: Concatenate X_features and y_true_target and save
print("\nStep 14: Saving final pre-draft feature set...")
final_predraft_df = pd.concat([X_features, y_true_target], axis=1)
try:
    final_predraft_df.to_parquet("wr_features_predraft_final.parquet", index=False)
    print("DataFrame saved successfully to wr_features_predraft_final.parquet.")
    print(f"Final saved shape: {final_predraft_df.shape}")
except Exception as e:
    print(f"Error saving DataFrame to parquet: {e}")
print("-" * 30)

print("Pre-draft feature engineering script finished.")
