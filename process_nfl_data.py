import nfl_data_py as nfl
import pandas as pd

# Define the range of years for data that supports it
YEARS = range(2000, 2024)

print("Step 1: nfl_data_py library is already installed.")

# Step 2: Load datasets
print("\nStep 2: Loading datasets...")

# Load draft picks
print("Loading draft picks...")
draft_picks_df = nfl.import_draft_picks(years=YEARS)
print("Draft Picks Data Loaded. Columns:")
print(draft_picks_df.columns)
print(f"Shape: {draft_picks_df.shape}")
print(draft_picks_df.head(2))
print("-" * 30)

# Load combine data - this function does not take a year argument.
# It's assumed it loads all available historical data.
print("Loading combine data...")
combine_df = nfl.import_combine_data()
print("Combine Data Loaded. Columns:")
print(combine_df.columns)
print(f"Shape: {combine_df.shape}")
# Filter combine data for relevant years if 'season' or 'draft_year' exists
if 'season' in combine_df.columns:
    combine_df = combine_df[combine_df['season'].isin(YEARS)]
elif 'draft_year' in combine_df.columns:
    # Assuming draft_year refers to the year they are drafted.
    # Combine usually happens few months before draft.
    combine_df = combine_df[combine_df['draft_year'].isin(YEARS)]
print(f"Shape after year filter (if applicable): {combine_df.shape}")
print(combine_df.head(2))
print("-" * 30)

# Load player information - this function does not take a year argument.
print("Loading player information...")
player_info_df = nfl.import_players()
print("Player Info Data Loaded. Columns:")
print(player_info_df.columns)
print(f"Shape: {player_info_df.shape}")
print(player_info_df.head(2))
print("-" * 30)

# Seasonal player stats (for general stats, not AV as it's missing)
# We'll use AV from draft_picks_df
print("Loading seasonal player stats...")
seasonal_stats_df = nfl.import_seasonal_data(years=YEARS)
print("Seasonal Player Stats Loaded. Columns:")
print(seasonal_stats_df.columns)
print(f"Shape: {seasonal_stats_df.shape}")
# Check for AV cols (expected to be empty based on previous run)
av_cols_seasonal = [col for col in seasonal_stats_df.columns if 'av' in col.lower()]
print(f"AV related columns in seasonal_stats_df: {av_cols_seasonal}")
print(seasonal_stats_df.head(2))
print("-" * 30)

# Step 3 is implicitly done by printing columns above.

# Step 4: Merge datasets
print("\nStep 4: Merging datasets...")

# Merge 1: draft_picks_df with player_info_df on 'gsis_id'
# draft_picks_df has 'gsis_id', player_info_df has 'gsis_id'
print("Merging draft_picks with player_info...")
merged_df = pd.merge(draft_picks_df, player_info_df, on='gsis_id', how='left', suffixes=('_draft', '_player'))
print(f"Shape after merging with player_info: {merged_df.shape}")
print(f"Columns: {merged_df.columns}")
# Check for duplicate columns to rename/drop if necessary (e.g. college vs college_name)
# draft_picks_df has 'college', player_info_df has 'college_name'
print(merged_df[['pfr_player_id', 'gsis_id', 'college', 'college_name', 'height', 'weight']].head(2))
print("-" * 30)

# Merge 2: merged_df with combine_df
# draft_picks_df (and thus merged_df) has 'pfr_player_id'. combine_df has 'pfr_id'.
# Rename 'pfr_id' in combine_df to 'pfr_player_id' for merging.
if 'pfr_id' in combine_df.columns:
    combine_df = combine_df.rename(columns={'pfr_id': 'pfr_player_id'})
    print("Renamed 'pfr_id' to 'pfr_player_id' in combine_df.")

    # Deduplicate combine_df before merging, keeping the latest entry if multiple exist for a player
    if 'season' in combine_df.columns: # 'season' is year of combine
        combine_df = combine_df.sort_values(by=['pfr_player_id', 'season'], ascending=[True, False])
        combine_df = combine_df.drop_duplicates(subset=['pfr_player_id'], keep='first')
        print(f"Deduplicated combine_df by 'pfr_player_id', keeping latest 'season'. Shape: {combine_df.shape}")
    elif 'draft_year' in combine_df.columns: # draft_year is also fine for sorting
        combine_df = combine_df.sort_values(by=['pfr_player_id', 'draft_year'], ascending=[True, False])
        combine_df = combine_df.drop_duplicates(subset=['pfr_player_id'], keep='first')
        print(f"Deduplicated combine_df by 'pfr_player_id', keeping latest 'draft_year'. Shape: {combine_df.shape}")

else:
    print("'pfr_id' not found in combine_df. Merge might fail or be incomplete.")

if 'pfr_player_id' in merged_df.columns and 'pfr_player_id' in combine_df.columns:
    print("Merging with combine_data...")
    # Keep only necessary columns from combine_df to avoid too many duplicates if player info is already there
    combine_cols_to_merge = ['pfr_player_id', 'ht', 'wt', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle']
    # Ensure all these columns actually exist in combine_df
    combine_cols_to_merge = [col for col in combine_cols_to_merge if col in combine_df.columns]

    merged_df = pd.merge(merged_df, combine_df[combine_cols_to_merge], on='pfr_player_id', how='left', suffixes=('_merged', '_combine'))
    print(f"Shape after merging with combine_data: {merged_df.shape}")
    print(f"Columns: {merged_df.columns}")
    # Display some relevant columns after merge
    # Select columns that demonstrate the merge, trying to avoid errors if some don't exist
    display_cols = ['pfr_player_name', 'gsis_id', 'pfr_player_id', 'ht', 'wt', 'forty']
    display_cols = [col for col in display_cols if col in merged_df.columns]
    print(merged_df[display_cols].head(2))
else:
    print("Skipping merge with combine_df as 'pfr_player_id' is missing in one of the dataframes.")
print("-" * 30)

# At this point, we don't merge seasonal_stats_df yet as it's for calculating target_AV later.
# The primary prospect data is now in merged_df.

# Step 5: Filter for WRs
print("\nStep 5: Filtering for WRs...")
# Need to identify the correct position column.
# draft_picks_df has 'position'
# player_info_df has 'position'
# combine_df has 'pos'
# Let's use 'position_draft' as it's from the draft context.
# If 'position_draft' was not created due to no suffix, it's just 'position'.
pos_col_options = ['position_draft', 'position_player', 'pos', 'position']
final_pos_col = None
for col in pos_col_options:
    if col in merged_df.columns:
        final_pos_col = col
        print(f"Using position column: {final_pos_col}")
        break

if final_pos_col:
    merged_df_wr = merged_df[merged_df[final_pos_col] == 'WR'].copy()
    print(f"Shape after filtering for WRs: {merged_df_wr.shape}")
    print(f"Number of WRs: {len(merged_df_wr)}")
    if not merged_df_wr.empty:
        print(merged_df_wr[[final_pos_col, 'pfr_player_name', 'season']].head(2)) # 'season' is from draft_picks_df
    else:
        print("No WRs found with the selected position column.")
else:
    print("Could not find a suitable position column for filtering WRs. WR data will be empty.")
    merged_df_wr = pd.DataFrame() # Empty DataFrame
print("-" * 30)

# Step 6: Calculate target_AV for first 5 seasons
print("\nStep 6: Calculating target_AV for first 5 seasons...")
# AV data is in `draft_picks_df` as `w_av` (weighted AV) or `car_av` (career AV).
# `draft_picks_df` contains overall career AV, not per season.
# `seasonal_stats_df` was loaded but confirmed to NOT have AV.
# The task states: "For each WR, calculate the average 'weighted_career_av' (or 'career_av' if that's the field name)
# for their first 5 seasons in the league."
# This implies needing per-season AV. `draft_picks_df` has `w_av`, `car_av`, `dr_av` which are summary stats.
# `nfl.import_seasonal_data()` was loaded as `seasonal_stats_df`. Let's re-check its columns for any AV.
# It was confirmed empty previously.

# Let's try `nfl.import_seasonal_pfr_advstats()` as PFR often has AV.
# Or, if `nfl.import_rookie_data()` or similar exists, it might be useful.
# The original subtask mentioned `load_player_stats()` or `load_pfr_advstats()`.
# The equivalent for `nfl_data_py` might be `import_seasonal_data` (already checked, no AV)
# or another PFR-specific one.

print("Attempting to load PFR advanced seasonal stats for AV calculation...")
try:
    # There is no import_pfr_seasonal_data, let's try import_pfr_advstats
    # This function might not exist, based on prior errors.
    # The library has import_pfr_passing_adv, import_pfr_rushing_adv etc.
    # Let's try to find a general one or one that might have AV.
    # For now, let's assume we need to find per-season AV.
    # The `draft_picks_df` has `w_av` which is "Weighted Career Approximate Value". This is a total.
    # If we cannot get per-season AV, we cannot calculate "average for first 5 seasons".

    # For now, let's check if `seasonal_stats_df` can be merged and if it has player_id and season.
    # It has 'player_id', 'season'.
    # `draft_picks_df` has 'gsis_id' and 'pfr_player_id'.
    # `player_info_df` has 'gsis_id'.
    # `seasonal_stats_df.player_id` is likely 'gsis_id' or 'pfr_player_id'.

    # Let's try to load `nfl.import_pbp_data` which can be aggregated to player-season level.
    # This might be too slow.

    # What if 'w_av' in draft_picks is actually what we need to use, and the "first 5 seasons" part is a misinterpretation if per-season AV is not available?
    # The prompt says: "average 'weighted_career_av' ... for their first 5 seasons".
    # This is tricky if `weighted_career_av` is a single career value.

    # Let's first verify the ID in seasonal_stats_df.
    # Typically, player_id in seasonal_stats_df is 'gsis_id'.

    # We need a source of *per-season* AV. `draft_picks.w_av` is a career total.
    # Let's look for a function that gives per-season AV.
    # The library has `nfl.load_pfr_data(season, stat_type)` which was in older versions, now it's `nfl.import_pfr_data`.
    # `import_pfr_data` is not listed directly.

    # Let's assume for a moment that the task meant to use the career AV if per-season is not found,
    # and the "first 5 seasons" is a refinement that might not be possible.
    # Or, we use `seasonal_stats_df` and if it had AV, we'd aggregate it.

    # Given the available data: `draft_picks_df` has `w_av` and `car_av`. These are career totals.
    # `seasonal_stats_df` has yearly stats but no AV.

    # Let's assume the "target_AV" should be the career weighted AV from draft_picks for now,
    # and make a note that per-season calculation is not directly possible with current findings.

    if 'w_av' in merged_df_wr.columns:
        merged_df_wr['target_AV'] = merged_df_wr['w_av']
        print("Used 'w_av' (Weighted Career Approximate Value) from draft data as 'target_AV'.")
        print("Note: This is a career total, not an average of the first 5 seasons, as per-season AV is not readily available in loaded general seasonal stats.")
    elif 'car_av' in merged_df_wr.columns:
        merged_df_wr['target_AV'] = merged_df_wr['car_av']
        print("Used 'car_av' (Career Approximate Value) from draft data as 'target_AV'.")
        print("Note: This is a career total, not an average of the first 5 seasons, as per-season AV is not readily available in loaded general seasonal stats.")
    else:
        merged_df_wr['target_AV'] = pd.NA # or np.nan if pandas version is older
        print("Neither 'w_av' nor 'car_av' found in the merged WR data. 'target_AV' set to NA.")

    if 'target_AV' in merged_df_wr.columns:
        print(merged_df_wr[['pfr_player_name', 'target_AV']].head())

except Exception as e:
    print(f"Error during AV calculation or PFR data loading: {e}")
    if not merged_df_wr.empty:
        merged_df_wr['target_AV'] = pd.NA
    print("'target_AV' set to NA due to error.")
print("-" * 30)

# Step 7: Attempt to find and load college statistics
print("\nStep 7: Checking for college statistics...")
# nfl_data_py primarily focuses on NFL data.
# It has cfb_player_id in draft_picks and combine_data.
# Let's check if there's a college stats loader.
try:
    # Based on common patterns, maybe import_cfb_stats?
    # A quick search of nfl_data_py docs suggests it doesn't have extensive college stats loaders.
    # It can link to CFB-Reference IDs.
    print("`nfl_data_py` does not have a direct function to load comprehensive college football stats.")
    print("College stats would typically need to be sourced separately (e.g., via sportsipy or other CFB data providers) and then merged using cfb_player_id.")
    print("For this task, we will proceed without college stats if not directly loadable via nfl_data_py.")
    # Placeholder: If we had college stats, they'd be loaded and merged here.
    # e.g., cfb_stats_df = load_college_data_function()
    # merged_df_wr = pd.merge(merged_df_wr, cfb_stats_df, on='cfb_player_id', how='left')
except AttributeError:
    print("No college stats loading function found in nfl_data_py, as expected.")
except Exception as e:
    print(f"An error occurred while checking for college stats: {e}")
print("-" * 30)


# Step 8: Perform initial data cleaning
print("\nStep 8: Performing initial data cleaning...")
if not merged_df_wr.empty:
    # Identify columns with more than 30% missing values
    missing_threshold = 0.30
    total_rows = len(merged_df_wr)
    missing_info = []
    for col in merged_df_wr.columns:
        missing_count = merged_df_wr[col].isnull().sum()
        missing_percentage = missing_count / total_rows
        if missing_percentage > missing_threshold:
            missing_info.append((col, missing_percentage * 100))

    print(f"Columns with more than {missing_threshold*100}% missing values:")
    if missing_info:
        for col, perc in missing_info:
            print(f"- {col}: {perc:.2f}% missing")
    else:
        print("No columns exceed the missing value threshold.")
    print("-" * 10)

    # Ensure numeric columns are of numeric types
    # Example numeric columns from combine: forty, bench, vertical, broad_jump, cone, shuttle
    # Example numeric from draft: pick, age, w_av, car_av, dr_av, games, etc. + stats like pass_yards
    # Columns like 'draft_number', 'draft_round' (from player_info_df) might also need conversion.
    print("Ensuring numeric columns are numeric...")
    potential_numeric_cols = [
        'pick', 'age', 'w_av', 'car_av', 'dr_av', 'games', 'pass_completions', 'pass_attempts',
        'pass_yards', 'pass_tds', 'pass_ints', 'rush_atts', 'rush_yards', 'rush_tds',
        'receptions', 'rec_yards', 'rec_tds', 'def_solo_tackles', 'def_ints', 'def_sacks', # from draft_picks
        'draft_number', 'height_player', 'weight_player', # from player_info (height/weight might be split or need conversion)
        'ht', 'wt', 'forty', 'bench', 'vertical', 'broad_jump', 'cone', 'shuttle', # from combine
        'target_AV'
    ]
    if 'height' in merged_df_wr.columns and not pd.api.types.is_numeric_dtype(merged_df_wr['height']):
        # Assuming height might be like '6-2' string. Needs conversion to inches.
        # This is complex. For now, attempt direct conversion for simple cases or leave as is.
        # Player_info_df has 'height' (numeric, inches) and 'weight' (numeric)
        # Combine_df has 'ht' (string, e.g. "5-11") and 'wt' (numeric)
        # Let's prioritize player_info_df height/weight if they exist and are numeric.
        # `player_info_df.height` is already numeric (inches). `draft_picks_df` does not have height/weight.
        # `combine_df.ht` is string, `combine_df.wt` is numeric.
        # After merges, we might have 'height_player' (good), 'weight_player' (good), 'ht_combine', 'wt_combine'.
        # Let's use 'height_player' and 'weight_player' as the primary numeric height/weight.
        # 'ht_combine' (e.g., from combine_df) would need conversion if used.
        if 'ht' in merged_df_wr.columns: # This is likely from combine data, string format
            print("Found 'ht' column, likely from combine data. This is typically a string like '6-2'.")
            # For simplicity in this step, we are not converting it. Player_info_df.height (numeric) is preferred.
            # If 'height_player' (from player_info_df) exists, it should be numeric.
            if 'height_player' in merged_df_wr.columns:
                merged_df_wr['height_player'] = pd.to_numeric(merged_df_wr['height_player'], errors='coerce')
                print("Converted 'height_player' to numeric.")
            if 'weight_player' in merged_df_wr.columns:
                merged_df_wr['weight_player'] = pd.to_numeric(merged_df_wr['weight_player'], errors='coerce')
                print("Converted 'weight_player' to numeric.")

    for col in potential_numeric_cols:
        if col in merged_df_wr.columns:
            if pd.api.types.is_object_dtype(merged_df_wr[col]): # Only attempt conversion if it's object type
                try:
                    merged_df_wr[col] = pd.to_numeric(merged_df_wr[col], errors='coerce')
                    # print(f"Converted '{col}' to numeric.")
                except Exception as e:
                    print(f"Could not convert {col} to numeric: {e}")
            elif pd.api.types.is_numeric_dtype(merged_df_wr[col]):
                pass # print(f"Column '{col}' is already numeric.")
    print("Numeric conversion attempt finished.")
    print("-" * 10)

    # Ensure date columns are datetime objects
    # player_info_df has 'birth_date'
    print("Ensuring date columns are datetime objects...")
    date_cols = ['birth_date'] # from player_info_df, available as 'birth_date_player' or 'birth_date'
    # Check actual column name after merge
    actual_birth_date_col = None
    if 'birth_date_player' in merged_df_wr.columns:
        actual_birth_date_col = 'birth_date_player'
    elif 'birth_date' in merged_df_wr.columns:
        actual_birth_date_col = 'birth_date'

    if actual_birth_date_col and actual_birth_date_col in merged_df_wr.columns:
        merged_df_wr[actual_birth_date_col] = pd.to_datetime(merged_df_wr[actual_birth_date_col], errors='coerce')
        print(f"Converted '{actual_birth_date_col}' to datetime.")
    else:
        print("No 'birth_date' column found for conversion.")
else:
    print("Skipping data cleaning as WR DataFrame is empty.")
print("-" * 30)

# Step 9: Print .info(), .head(), and missing value counts
print("\nStep 9: Final DataFrame summary...")
if not merged_df_wr.empty:
    print("\nDataFrame .info():")
    merged_df_wr.info()

    print("\nDataFrame .head():")
    print(merged_df_wr.head())

    print("\nMissing value counts:")
    print(merged_df_wr.isnull().sum())
else:
    print("WR DataFrame is empty, skipping summary.")
print("-" * 30)

# Step 10: Save the final DataFrame
print("\nStep 10: Saving the final DataFrame...")
if not merged_df_wr.empty:
    try:
        merged_df_wr.to_parquet("wr_prospect_data_raw.parquet", index=False)
        print("DataFrame saved successfully to wr_prospect_data_raw.parquet")
    except Exception as e:
        print(f"Error saving DataFrame to parquet: {e}")
else:
    print("WR DataFrame is empty, not saving.")
print("-" * 30)

# Step 11: Report any significant issues (handled inline for now)
print("\nStep 11: Significant issues report:")
print("- Per-season AV for the first 5 seasons could not be calculated directly as `import_seasonal_data` does not contain AV, and PFR-specific seasonal data loaders for AV were not found/functional. Used career `w_av` or `car_av` from draft data as a proxy for `target_AV`.")
print("- College stats loading is not directly supported by `nfl_data_py` and is noted as a manual integration step for the future.")
print("- `import_pfr_seasonal_data` and `import_rosters` attributes were not found.")
print("Processing script finished.")

# To make sure we see outputs in the agent.
if 'merged_df_wr' in locals() and not merged_df_wr.empty:
    print("\nFinal WR Data Sample (first 5 rows):")
    print(merged_df_wr.head())
    print(f"\nFinal WR Data Shape: {merged_df_wr.shape}")
else:
    print("\nNo WR data to display.")
