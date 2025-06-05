import pandas as pd
import numpy as np

# Step 1: Load wr_prospect_data_college.parquet
print("Step 1: Loading wr_prospect_data_college.parquet...")
try:
    df = pd.read_parquet("wr_prospect_data_college.parquet")
    print("Successfully loaded wr_prospect_data_college.parquet.")
except FileNotFoundError:
    print("Error: wr_prospect_data_college.parquet not found. Please ensure this file exists.")
    exit()
except Exception as e:
    print(f"Error loading wr_prospect_data_college.parquet: {e}")
    exit()

# Step 2: Print initial DataFrame info and missing values
print("\nStep 2: Initial DataFrame Info & Missing Values...")
print("DataFrame .info():")
df.info(verbose=True, show_counts=True)
print("\nMissing value counts (initial):")
print(df.isnull().sum())
print("-" * 30)

# Step 3: Drop car_av column if it exists
print("\nStep 3: Drop 'car_av' column...")
if 'car_av' in df.columns:
    df = df.drop(columns=['car_av'])
    print("'car_av' column dropped.")
else:
    print("'car_av' column not found.")
print("-" * 30)

# Step 4: Calculate age_at_draft
print("\nStep 4: Calculate 'age_at_draft'...")
# Assuming 'season' column is the draft year (as established in previous tasks from nfl_data_py.import_draft_picks)
# And 'birth_date' is the column from nfl_data_py.import_players
if 'season' in df.columns and 'birth_date' in df.columns:
    df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
    # Calculate age. Draft day is typically April. For simplicity, use draft year - birth year.
    # A more precise calculation would use month/day of draft vs birth month/day.
    df['age_at_draft'] = df['season'] - df['birth_date'].dt.year
    print("Calculated 'age_at_draft'.")
    print(df[['season', 'birth_date', 'age_at_draft']].head())
    # Check for issues like draft year being before birth year (results in negative age)
    if (df['age_at_draft'] < 18).any(): # Assuming no one is drafted younger than 18
        print("Warning: Some 'age_at_draft' values are unusually low or negative.")
        print(df[df['age_at_draft'] < 18][['pfr_player_name', 'season', 'birth_date', 'age_at_draft']])
else:
    print("Could not calculate 'age_at_draft': 'season' or 'birth_date' column missing.")
print("-" * 30)

# Step 5: Calculate BMI
print("\nStep 5: Calculate 'bmi'...")
# Height from nfl_data_py.import_players is typically in inches, stored in 'height' column.
# Weight from nfl_data_py.import_players is typically in lbs, stored in 'weight' column.
# Combine data also has 'ht' (string 'feet-inches') and 'wt' (lbs).
# We should prioritize the direct 'height' and 'weight' columns if they are numeric and well-populated.
# From previous subtask, 'height' and 'weight' (from player_info merge) were numeric and populated.
if 'height' in df.columns and 'weight' in df.columns:
    # Ensure they are numeric
    df['height'] = pd.to_numeric(df['height'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

    # BMI = (Weight in Pounds / (Height in inches x Height in inches)) x 703
    # Avoid division by zero or NaN height/weight
    df['bmi'] = np.where(
        (df['height'].notna() & df['weight'].notna() & (df['height'] > 0)),
        (df['weight'] / (df['height']**2)) * 703,
        np.nan
    )
    print("Calculated 'bmi'.")
    print(df[['pfr_player_name', 'height', 'weight', 'bmi']].head())
else:
    print("Could not calculate 'bmi': 'height' or 'weight' column missing.")
print("-" * 30)

# Step 6: Handle Missing Values
print("\nStep 6: Handling Missing Values...")
# Identify combine-related columns
# Common names: 'forty' (40-yard dash), 'vertical' (vertical jump), 'bench' (bench press),
# 'broad_jump', 'cone' (3-cone drill), 'shuttle' (20-yard shuttle).
combine_cols = []
if 'forty' in df.columns: combine_cols.append('forty')
if 'vertical' in df.columns: combine_cols.append('vertical')
if 'bench' in df.columns: combine_cols.append('bench') # Bench press reps
if 'broad_jump' in df.columns: combine_cols.append('broad_jump')
if 'cone' in df.columns: combine_cols.append('cone')
if 'shuttle' in df.columns: combine_cols.append('shuttle')
# ht and wt from combine data were also loaded, but we are prioritizing 'height' and 'weight' from player_info.
# If 'ht' (combine height, string) and 'wt' (combine weight) are to be used, 'ht' needs conversion.
# For now, focusing on the performance metrics.

print(f"Identified combine columns for imputation: {combine_cols}")
for col in combine_cols:
    if df[col].isnull().any():
        # Create indicator column
        df[f'{col}_missing'] = df[col].isnull().astype(int)
        # Impute with median
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed missing values in '{col}' with median ({median_val}) and created '{col}_missing' indicator.")
    else:
        df[f'{col}_missing'] = 0 # If no missing values, indicator is all 0s
        print(f"No missing values in '{col}'. Created '{col}_missing' indicator with all 0s.")

# For 'college_name', if missing, impute with 'Unknown'.
# The 'college' column from draft_picks was used. Let's assume this is the primary one.
# player_info also has 'college_name'. If distinct, one should have been chosen or coalesced.
# Based on previous script, 'college' from draft_picks was primary, and 'college_name' from player_info was also present.
# Let's assume 'college' is the main one to use for this step.
main_college_col = 'college' # This was from draft_picks
if main_college_col not in df.columns and 'college_name' in df.columns:
    main_college_col = 'college_name' # Fallback if 'college' isn't there for some reason
elif main_college_col not in df.columns:
    main_college_col = None
    print("Warning: Neither 'college' nor 'college_name' found for imputation.")

if main_college_col and df[main_college_col].isnull().any():
    df[main_college_col].fillna('Unknown', inplace=True)
    print(f"Imputed missing values in '{main_college_col}' with 'Unknown'.")

# Report on any other columns with missing data and how they were handled
print("\nChecking other numeric columns for missing values and applying median imputation:")
# Example: 'age_at_draft', 'bmi', 'w_av', 'dr_av', 'games' (from draft picks), etc.
# 'target_AV' is crucial, should check its missingness but imputation strategy might differ (or be deferred).
numeric_cols_to_check_impute = ['age_at_draft', 'bmi', 'w_av', 'dr_av', 'games', 'seasons_started',
                                'allpro', 'probowls', 'pass_completions', 'pass_attempts', 'pass_yards',
                                'pass_tds', 'pass_ints', 'rush_atts', 'rush_yards', 'rush_tds',
                                'receptions', 'rec_yards', 'rec_tds', 'def_solo_tackles', 'def_ints', 'def_sacks']
# Add college stats if they exist (they won't from previous run, but for robustness)
college_numeric_cols = ['college_games_total', 'college_receptions_total', 'college_yards_total', 'college_tds_total']
numeric_cols_to_check_impute.extend([col for col in college_numeric_cols if col in df.columns])


for col in numeric_cols_to_check_impute:
    if col in df.columns and df[col].isnull().any():
        if col != 'target_AV': # Don't impute target_AV with simple median yet
            df[f'{col}_missing'] = df[col].isnull().astype(int)
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Imputed missing values in '{col}' with median ({median_val}) and created '{col}_missing' indicator.")
        else:
            print(f"Column '{col}' has missing values but will not be imputed at this stage.")
    elif col in df.columns: # No missing values
         df[f'{col}_missing'] = 0


# For 'target_AV', if it has missing values, create an indicator but do not impute yet.
if 'target_AV' in df.columns and df['target_AV'].isnull().any():
    df['target_AV_missing'] = df['target_AV'].isnull().astype(int)
    print("Created 'target_AV_missing' indicator.")
elif 'target_AV' in df.columns:
    df['target_AV_missing'] = 0
    print("No missing values in 'target_AV'. Created 'target_AV_missing' indicator with all 0s.")


print("\nRemaining missing values after initial imputation pass:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("-" * 30)


# Step 7: Categorical Feature Encoding
print("\nStep 7: Categorical Feature Encoding...")
if main_college_col and main_college_col in df.columns:
    print(f"Encoding '{main_college_col}' using frequency encoding...")
    college_freq = df[main_college_col].value_counts(normalize=True)
    df['college_freq_encoded'] = df[main_college_col].map(college_freq)
    print(f"'{main_college_col}' frequency encoded into 'college_freq_encoded'.")
    print(df[[main_college_col, 'college_freq_encoded']].head())

    # Drop original college_name column (if it's not the only one, e.g. if 'college' and 'college_name' both exist)
    # For now, assume main_college_col is the one to drop after encoding.
    # If 'college' and 'college_name' are different and both exist, this might need refinement.
    # The setup from nfl_data_py usually makes 'college' (from draft) and 'college_name' (from player profile) distinct.
    # The previous step imputed main_college_col. If 'college' and 'college_name' are different,
    # the other one might still have NaNs.

    # Let's be specific: drop the one we used for encoding.
    df = df.drop(columns=[main_college_col])
    print(f"Dropped original '{main_college_col}' column.")
    # If 'college_name' also exists and is different from main_college_col, consider dropping it too or encoding it.
    # For this pass, just dropping the encoded one.
    if main_college_col == 'college' and 'college_name' in df.columns:
        print("Note: 'college_name' column also exists. Consider handling or dropping it if it's redundant or not needed.")
    elif main_college_col == 'college_name' and 'college' in df.columns:
         print("Note: 'college' column also exists. Consider handling or dropping it if it's redundant or not needed.")

else:
    print("No college column found or specified for frequency encoding.")
print("-" * 30)

# Step 8: Final Feature Selection & Cleanup
print("\nStep 8: Final Feature Selection & Cleanup...")
cols_to_drop = []
# Identifiers
if 'pfr_player_id' in df.columns: cols_to_drop.append('pfr_player_id')
if 'gsis_id' in df.columns: cols_to_drop.append('gsis_id') # Often used as ID
if 'cfb_player_id' in df.columns: cols_to_drop.append('cfb_player_id') # College ID
if 'esb_id' in df.columns: cols_to_drop.append('esb_id')
if 'gsis_it_id' in df.columns: cols_to_drop.append('gsis_it_id')
if 'smart_id' in df.columns: cols_to_drop.append('smart_id')

# Name columns (use pfr_player_name for reference if needed, but drop for model training)
name_cols = ['pfr_player_name', 'display_name', 'first_name', 'last_name', 'short_name', 'football_name']
for col in name_cols:
    if col in df.columns:
        cols_to_drop.append(col)

if 'birth_date' in df.columns: cols_to_drop.append('birth_date') # Used for age_at_draft

# Other text/object columns that are not features or have been encoded
if 'team' in df.columns: cols_to_drop.append('team') # Draft team abbreviation
if 'category' in df.columns: cols_to_drop.append('category') # Draft category
if 'side' in df.columns: cols_to_drop.append('side') # Draft side (e.g. offense)
if 'college_conference' in df.columns: cols_to_drop.append('college_conference')
if 'current_team_id' in df.columns: cols_to_drop.append('current_team_id')
if 'draft_club' in df.columns: cols_to_drop.append('draft_club') # Redundant with 'team'?
if 'headshot' in df.columns: cols_to_drop.append('headshot')
if 'status' in df.columns: cols_to_drop.append('status')
if 'status_description_abbr' in df.columns: cols_to_drop.append('status_description_abbr')
if 'status_short_description' in df.columns: cols_to_drop.append('status_short_description')
if 'team_abbr' in df.columns: cols_to_drop.append('team_abbr') # likely same as 'team'
if 'uniform_number' in df.columns: cols_to_drop.append('uniform_number')
if 'years_of_experience' in df.columns: cols_to_drop.append('years_of_experience') # Object type, might need parsing if useful
if 'suffix' in df.columns: cols_to_drop.append('suffix')
if 'college_name' in df.columns and 'college_name' not in cols_to_drop and main_college_col != 'college_name':
    cols_to_drop.append('college_name') # Drop if it wasn't the main encoded one and still exists

# College stats source columns if they exist from previous step (they won't if college data was empty)
if 'cfbd_player_name' in df.columns: cols_to_drop.append('cfbd_player_name')
if 'cfbd_college_name' in df.columns: cols_to_drop.append('cfbd_college_name')
if 'college_stat_source' in df.columns: cols_to_drop.append('college_stat_source')


# Position column - check if it's constant for WRs
pos_col_to_check = None
if 'position_draft' in df.columns: pos_col_to_check = 'position_draft'
elif 'position_player' in df.columns: pos_col_to_check = 'position_player'
elif 'position' in df.columns: pos_col_to_check = 'position'

if pos_col_to_check and df[pos_col_to_check].nunique() == 1 and df[pos_col_to_check].iloc[0] == 'WR':
    print(f"Dropping '{pos_col_to_check}' as it's constant ('WR').")
    cols_to_drop.append(pos_col_to_check)
    # Also drop other position columns if they exist
    other_pos_cols = ['position_draft', 'position_player', 'position', 'pos', 'position_group']
    for opc in other_pos_cols:
        if opc in df.columns and opc not in cols_to_drop:
             cols_to_drop.append(opc)
elif pos_col_to_check:
    print(f"Position column '{pos_col_to_check}' is not constant or not 'WR'. It will be kept or needs one-hot encoding if categorical.")
    # For now, if not constant WR, let's add it to drop list and handle encoding later if needed.
    # This simplifies the current step.
    # cols_to_drop.append(pos_col_to_check)

# Remove duplicates from cols_to_drop and ensure they exist in df
final_cols_to_drop = list(set(col for col in cols_to_drop if col in df.columns))
df = df.drop(columns=final_cols_to_drop)
print(f"Dropped columns: {final_cols_to_drop}")

# Ensure target_AV is present
if 'target_AV' not in df.columns:
    print("CRITICAL WARNING: 'target_AV' is missing from the DataFrame after cleanup.")
else:
    print("'target_AV' is present.")
print("-" * 30)

# Step 9: Print .info(), .head(), and missing value counts for the final DataFrame
print("\nStep 9: Final DataFrame Info & Missing Values...")
print("DataFrame .info():")
df.info(verbose=True, show_counts=True)
print("\nMissing value counts (final):")
print(df.isnull().sum()[df.isnull().sum() > 0]) # Only show columns with missing values
print("\nDataFrame .head():")
print(df.head())
print("-" * 30)

# Step 10: Save the DataFrame as wr_features.parquet
print("\nStep 10: Saving DataFrame to wr_features.parquet...")
try:
    df.to_parquet("wr_features.parquet", index=False)
    print("DataFrame saved successfully to wr_features.parquet.")
except Exception as e:
    print(f"Error saving DataFrame to parquet: {e}")
print("-" * 30)

print("Feature engineering script finished.")
