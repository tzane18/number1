import pandas as pd
import numpy as np
import time

# Step 1: Define years
PBP_YEARS = range(2014, 2023)
ROSTER_YEARS = range(2014, 2023)

# Step 2: Initialize lists
all_pbp_dfs = []
all_roster_dfs = []
new_college_stat_cols = ['college_games_played', 'college_receptions', 'college_receiving_yards', 'college_receiving_tds', 'college_data_source']

base_urls = [
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/",
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/master/"
]

# --- Download Roster Data ---
print("Attempting to download Roster data...")
roster_path_templates = [
    ("rosters/parquet/rosters_{year}.parquet", "rosters/csv/rosters_{year}.csv.gz"),
    ("data/rosters/{year}/roster.parquet", "data/rosters/{year}/roster.csv.gz"), # From cfbfastR docs
]
roster_df_combined = pd.DataFrame()
first_roster_cols_printed = False

for year in ROSTER_YEARS:
    print(f"\nAttempting Roster for year: {year}")
    roster_df_year = None
    roster_found_for_year = False
    for base_url in base_urls:
        if roster_found_for_year: break
        for parquet_template, csv_template in roster_path_templates:
            if roster_found_for_year: break
            url = base_url + parquet_template.format(year=year)
            print(f"  Trying Roster Parquet: {url}")
            try:
                roster_df_year = pd.read_parquet(url)
                print(f"  Loaded Roster Parquet for {year} from {url}")
                roster_found_for_year = True
            except Exception:
                print(f"  Failed Roster Parquet.")
                url = base_url + csv_template.format(year=year)
                print(f"  Trying Roster CSV: {url}")
                try:
                    roster_df_year = pd.read_csv(url, compression='gzip', low_memory=False)
                    print(f"  Loaded Roster CSV for {year} from {url}")
                    roster_found_for_year = True
                except Exception:
                    print(f"  Failed Roster CSV.")

    if roster_df_year is not None and not roster_df_year.empty:
        if 'season' not in roster_df_year.columns and 'year' in roster_df_year.columns:
             roster_df_year.rename(columns={'year': 'season'}, inplace=True)
        if 'season' not in roster_df_year.columns: roster_df_year['season'] = year # Ensure season column
        all_roster_dfs.append(roster_df_year)
        if not first_roster_cols_printed:
            print(f"FIRST Roster DataFrame (year {year}) COLUMNS: {roster_df_year.columns.tolist()}")
            first_roster_cols_printed = True
    else:
        print(f"  No Roster data found for year {year}.")

if all_roster_dfs:
    roster_df_combined = pd.concat(all_roster_dfs, ignore_index=True)
    print(f"\nSuccessfully concatenated roster data. Shape: {roster_df_combined.shape}")
else:
    print("\nNo roster data downloaded. College team information will be missing for PBP aggregation.")
print("-" * 30)

# --- Download PBP (Player Stats) Data ---
print("\nAttempting to download PBP-like (player_stats) data...")
pbp_path_templates = [
    ("player_stats/parquet/player_stats_{year}.parquet", "player_stats/csv/player_stats_{year}.csv.gz"),
]
first_pbp_cols_printed = False

for year in PBP_YEARS:
    print(f"\nAttempting PBP-like data for year: {year}")
    year_df = None
    found_for_year = False
    for base_url in base_urls: # Prioritize main branch
        if found_for_year: break
        for parquet_template, csv_template in pbp_path_templates:
            if found_for_year: break
            url = base_url + parquet_template.format(year=year)
            print(f"  Trying PBP Parquet: {url}")
            try:
                year_df = pd.read_parquet(url)
                print(f"  Loaded PBP Parquet for {year} from {url}")
                found_for_year = True
            except Exception as e:
                print(f"  Failed PBP Parquet: {e}")
                url = base_url + csv_template.format(year=year)
                print(f"  Trying PBP CSV: {url}")
                try:
                    year_df = pd.read_csv(url, compression='gzip', low_memory=False)
                    print(f"  Loaded PBP CSV for {year} from {url}")
                    found_for_year = True
                except Exception as e2:
                    print(f"  Failed PBP CSV: {e2}")

    if year_df is not None and not year_df.empty:
        if 'season' not in year_df.columns and 'year' in year_df.columns:
             year_df.rename(columns={'year': 'season'}, inplace=True)
        elif 'season' not in year_df.columns: year_df['season'] = year
        all_pbp_dfs.append(year_df)
        if not first_pbp_cols_printed:
             print(f"FIRST PBP-like DataFrame (year {year}) COLUMNS: {year_df.columns.tolist()}")
             first_pbp_cols_printed = True
    else:
        print(f"  No PBP-like data found for year {year}.")

college_pbp_df = pd.DataFrame()
if all_pbp_dfs:
    college_pbp_df = pd.concat(all_pbp_dfs, ignore_index=True)
    print(f"\nSuccessfully concatenated PBP-like data. Shape: {college_pbp_df.shape}")
else:
    print("\nNo PBP-like data downloaded. Cannot proceed with college stat aggregation.")
print("-" * 30)


# Step 6: Aggregate PBP data to seasonal player stats and merge with Roster
college_seasonal_stats_df = pd.DataFrame()
if not college_pbp_df.empty:
    print("\nStep 6: Aggregating PBP data to seasonal player stats...")

    # Standardize play ID column name (some years use 'id_play', others 'play_id')
    if 'id_play' in college_pbp_df.columns and 'play_id' not in college_pbp_df.columns:
        college_pbp_df.rename(columns={'id_play': 'play_id'}, inplace=True)

    # Filter for plays with a reception
    receiving_plays = college_pbp_df[college_pbp_df['reception_player_id'].notna()].copy()
    receiving_plays['reception_player_id'] = receiving_plays['reception_player_id'].astype(str) # Ensure consistent ID type

    # Calculate receiving TDs
    receiving_plays['is_receiving_td'] = np.where(
        (receiving_plays['touchdown_player_id'].notna()) & \
        (receiving_plays['touchdown_player_id'] == receiving_plays['reception_player_id']) & \
        (receiving_plays['reception_yds'].notna()), # Ensure it was a reception leading to TD
        1, 0
    )

    # Aggregate stats
    agg_functions = {
        'reception_player': 'first', # Assumes name is consistent for player_id
        'game_id': pd.Series.nunique,
        'reception_yds': 'sum',
        'play_id': 'count', # Counts receptions
        'is_receiving_td': 'sum'
    }

    player_seasonal_stats_df = receiving_plays.groupby(
        ['season', 'reception_player_id'] # Group by season and player ID
    ).agg(agg_functions).reset_index()

    player_seasonal_stats_df.rename(columns={
        'reception_player_id': 'player_id',
        'reception_player': 'player_name',
        'game_id': 'college_games_played',
        'reception_yds': 'college_receiving_yards',
        'play_id': 'college_receptions',
        'is_receiving_td': 'college_receiving_tds'
    }, inplace=True)

    print(f"Aggregated seasonal stats. Shape: {player_seasonal_stats_df.shape}")

    if not roster_df_combined.empty:
        print("Merging aggregated stats with roster data...")
        roster_df_to_merge = roster_df_combined[['athlete_id', 'season', 'team']].copy()
        # Clean and standardize player_id types BEFORE renaming and merging
        # Convert PBP player_id (from reception_player_id)
        player_seasonal_stats_df['player_id'] = pd.to_numeric(player_seasonal_stats_df['player_id'], errors='coerce').astype('Int64').astype(str).replace('<NA>', np.nan)
        # Convert roster athlete_id
        roster_df_to_merge['athlete_id'] = pd.to_numeric(roster_df_to_merge['athlete_id'], errors='coerce').astype('Int64').astype(str).replace('<NA>', np.nan)

        roster_df_to_merge.rename(columns={'athlete_id': 'player_id', 'team': 'college_team_name'}, inplace=True)
        roster_df_to_merge.drop_duplicates(subset=['player_id', 'season'], keep='first', inplace=True)

        # Debugging the merge between aggregated PBP and Roster
        print("\nDebugging PBP stats and Roster merge:")
        print(f"PBP Aggregated Stats - sample IDs/seasons (pre-merge): {player_seasonal_stats_df[['player_id', 'season']].dropna(subset=['player_id']).head().to_dict('records')}")
        print(f"PBP Aggregated Stats - player_id dtype: {player_seasonal_stats_df['player_id'].dtype}, season dtype: {player_seasonal_stats_df['season'].dtype}")

        print(f"Roster Data - sample IDs/seasons (pre-merge): {roster_df_to_merge[['player_id', 'season']].dropna(subset=['player_id']).head().to_dict('records')}")
        print(f"Roster Data - player_id dtype: {roster_df_to_merge['player_id'].dtype}, season dtype: {roster_df_to_merge['season'].dtype}")

        # Ensure season is int for both and handle potential NaNs in player_id before merge
        player_seasonal_stats_df['season'] = player_seasonal_stats_df['season'].astype(int)
        roster_df_to_merge['season'] = roster_df_to_merge['season'].astype(int)
        player_seasonal_stats_df.dropna(subset=['player_id'], inplace=True)
        roster_df_to_merge.dropna(subset=['player_id'], inplace=True)


        # Perform an inner merge for debugging to see what matches
        debug_merge_df = pd.merge(
            player_seasonal_stats_df,
            roster_df_to_merge,
            on=['player_id', 'season'],
            how='inner' # Changed to inner for debugging
        )
        print(f"Shape after INNER merging PBP stats with roster: {debug_merge_df.shape}")
        if not debug_merge_df.empty:
            print("Sample of successful INNER merge with roster:")
            print(debug_merge_df[['player_id', 'season', 'player_name', 'college_team_name']].head())
        else:
            print("INNER merge with roster resulted in an empty DataFrame. No common player_id/season pairs.")

        # Original left merge
        player_seasonal_stats_df = pd.merge(
            player_seasonal_stats_df,
            roster_df_to_merge,
            on=['player_id', 'season'],
            how='left'
        )
        print(f"Shape after LEFT merging PBP stats with roster: {player_seasonal_stats_df.shape}")
        print("Columns after roster merge:", player_seasonal_stats_df.columns.tolist())
        # Check how many have non-null team names after left merge
        print(f"Number of rows with non-null college_team_name after merge: {player_seasonal_stats_df['college_team_name'].notna().sum()}")
        if player_seasonal_stats_df['college_team_name'].notna().any():
            print(player_seasonal_stats_df[player_seasonal_stats_df['college_team_name'].notna()].head())
        else:
            print("No college_team_name values were successfully merged from roster.")


    else:
        print("Roster data is empty, cannot merge for team names. `college_team_name` will be missing.")
        player_seasonal_stats_df['college_team_name'] = pd.NA # Add empty column if no roster

    # Normalize names for merging
    player_seasonal_stats_df['name_normalized'] = player_seasonal_stats_df['player_name'].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()
    player_seasonal_stats_df['team_normalized'] = player_seasonal_stats_df['college_team_name'].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()
    team_standardization_map = {'louisiana state': 'lsu', 'southern california': 'usc', 'southern methodist': 'smu'} # Add more
    player_seasonal_stats_df['team_normalized'] = player_seasonal_stats_df['team_normalized'].replace(team_standardization_map)

else:
    print("PBP data is empty. Skipping aggregation.")
college_df = player_seasonal_stats_df # Use this as the final college_df
print("-" * 30)


# Step 7: Load NFL prospect data
print("\nStep 7: Loading NFL prospect data (wr_prospect_data_college.parquet)...")
try:
    nfl_df = pd.read_parquet("wr_prospect_data_college.parquet")
    print(f"Successfully loaded NFL prospect data. Shape: {nfl_df.shape}")
except Exception as e:
    print(f"Error loading wr_prospect_data_college.parquet: {e}. Cannot proceed.")
    nfl_df = pd.DataFrame()
print("-" * 30)

# Step 8: Preprocess NFL prospect data for merging
print("\nStep 8: Preprocessing NFL prospect data for merging...")
if not nfl_df.empty:
    nfl_player_name_col = 'pfr_player_name' if 'pfr_player_name' in nfl_df.columns else 'player_name'
    nfl_college_name_col = 'college' if 'college' in nfl_df.columns else 'college_name'

    if nfl_player_name_col in nfl_df.columns:
        nfl_df['nfl_name_normalized'] = nfl_df[nfl_player_name_col].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()
    else: nfl_df['nfl_name_normalized'] = pd.NA

    if nfl_college_name_col in nfl_df.columns:
        nfl_df['nfl_team_normalized'] = nfl_df[nfl_college_name_col].astype(str).str.lower().str.replace('[^\w\s]', '', regex=True).str.strip()
        team_standardization_map = {'louisiana state': 'lsu', 'southern california': 'usc', 'southern methodist': 'smu'}
        nfl_df['nfl_team_normalized'] = nfl_df['nfl_team_normalized'].replace(team_standardization_map)
    else: nfl_df['nfl_team_normalized'] = pd.NA
else:
    print("NFL DataFrame is empty, skipping preprocessing.")
print("-" * 30)

# Step 9: Merge aggregated college_df with NFL prospect DataFrame
print("\nStep 9: Merging college stats into NFL prospect DataFrame...")
merged_nfl_df = nfl_df.copy()
num_prospects_merged_with_college_data = 0

if not college_df.empty and not nfl_df.empty and 'season' in merged_nfl_df.columns:
    for col in new_college_stat_cols: merged_nfl_df[col] = 0
    merged_nfl_df['college_data_source'] = pd.NA # Explicitly NA initially

    for index, prospect_row in merged_nfl_df.iterrows():
        prospect_name_norm = prospect_row['nfl_name_normalized']
        prospect_college_norm = prospect_row['nfl_team_normalized']
        target_college_season = prospect_row['season'] - 1 # NFL draft year - 1

        match = college_df[
            (college_df['name_normalized'] == prospect_name_norm) &
            (college_df['team_normalized'] == prospect_college_norm) &
            (college_df['season'] == target_college_season)
        ]

        if not match.empty:
            matched_stats = match.iloc[0]
            merged_nfl_df.loc[index, 'college_games_played'] = matched_stats.get('college_games_played', 0)
            merged_nfl_df.loc[index, 'college_receptions'] = matched_stats.get('college_receptions', 0)
            merged_nfl_df.loc[index, 'college_receiving_yards'] = matched_stats.get('college_receiving_yards', 0)
            merged_nfl_df.loc[index, 'college_receiving_tds'] = matched_stats.get('college_receiving_tds', 0)
            merged_nfl_df.loc[index, 'college_data_source'] = 'sportsdataverse_pbp'
            num_prospects_merged_with_college_data += 1
    print(f"Successfully merged college data for {num_prospects_merged_with_college_data} / {len(merged_nfl_df)} NFL prospects.")
    merged_nfl_df.drop(columns=['nfl_name_normalized', 'nfl_team_normalized'], inplace=True, errors='ignore')
else:
    print("Skipping merge: College data or NFL data is empty or missing key columns.")
print("-" * 30)

# Step 10: Save
print("\nStep 10: Saving the final DataFrame...")
output_filename = "wr_data_with_college.parquet"
try:
    merged_nfl_df.to_parquet(output_filename, index=False)
    print(f"DataFrame saved successfully to {output_filename}")
    if num_prospects_merged_with_college_data == 0:
        print("WARNING: No new college data was successfully merged for any prospect.")
except Exception as e:
    print(f"Error saving DataFrame: {e}")
print("-" * 30)

# Step 11: Print .info() and .head()
print("\nStep 11: Final DataFrame summary...")
if not merged_nfl_df.empty:
    print("DataFrame .info():")
    merged_nfl_df.info(verbose=True, show_counts=True)
    print("\nDataFrame .head() with new college columns (if any):")
    cols_to_show = ['pfr_player_name', 'college'] + [col for col in new_college_stat_cols if col in merged_nfl_df.columns]
    print(merged_nfl_df[cols_to_show].head())
else:
    print("Final DataFrame is empty.")
print("-" * 30)

print("College data download and merge script finished.")
