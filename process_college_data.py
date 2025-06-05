import pandas as pd
import cfbd
import os
import time

# Step 1: cfbd library installed in Turn 13.

# Step 2: Load the wr_prospect_data_raw.parquet file
print("Step 2: Loading wr_prospect_data_raw.parquet...")
try:
    nfl_prospects_df = pd.read_parquet("wr_prospect_data_raw.parquet")
    print("Successfully loaded wr_prospect_data_raw.parquet.")
    print(f"NFL Prospects DataFrame shape: {nfl_prospects_df.shape}")
except FileNotFoundError:
    print("Error: wr_prospect_data_raw.parquet not found. Please ensure this file exists from the previous subtask.")
    exit()
except Exception as e:
    print(f"Error loading wr_prospect_data_raw.parquet: {e}")
    exit()
print("-" * 30)

# Configure CFBD API
print("Configuring cfbd API...")
api_key = os.getenv("CFBD_API_KEY")
configuration = cfbd.Configuration()

if api_key:
    print("CFBD_API_KEY found in environment.")
    configuration.api_key['Authorization'] = api_key
    configuration.api_key_prefix['Authorization'] = 'Bearer'
else:
    print("CFBD_API_KEY not found in environment. API calls will likely fail or return limited data.")

api_client = cfbd.ApiClient(configuration)
players_api = cfbd.PlayersApi(api_client)
# teams_api = cfbd.TeamsApi(api_client) # For testing if needed later

# Test API call (optional, but good for seeing if key works at all)
# try:
#     print("Attempting test API call (get_teams)...")
#     test_teams = teams_api.get_teams(year=2022)
#     if test_teams:
#         print("Test API call successful.")
# except cfbd.ApiException as e:
#     print(f"Test API call failed: {e}. This may indicate an issue with the API key or network.")
# except Exception as e_gen:
#     print(f"Generic error during test API call: {e_gen}")
# print("-" * 30)


# Step 3 & 4: Iterate through prospects and fetch college stats (Simplified to one year)
print("\nStep 3 & 4: Fetching college stats for each prospect (simplified to one year)...")
college_stats_list = []
required_cols = ['pfr_player_name', 'college', 'season'] # 'season' is NFL draft year
missing_cols = [col for col in required_cols if col not in nfl_prospects_df.columns]

if missing_cols:
    print(f"Error: Missing required columns in nfl_prospects_df: {missing_cols}")
    exit()

prospects_to_process = nfl_prospects_df
total_prospects = len(prospects_to_process)
prospects_with_some_college_data = 0
api_key_seems_to_work = True # Assume true, set to false on 401

for index, row in prospects_to_process.iterrows():
    if not api_key_seems_to_work:
        print("Halting further API calls due to prior 401 error.")
        break

    player_name = row['pfr_player_name']
    nfl_college_name = row['college']
    draft_year = row['season']

    print(f"\nProcessing: {player_name}, College: {nfl_college_name}, Draft Year: {draft_year}")

    if pd.isna(player_name) or pd.isna(nfl_college_name):
        print(f"  Skipping {player_name} due to missing name or college in NFL data.")
        continue

    cfbd_player_id = None
    player_data_for_year = [] # Stores stats for the single year we fetch

    # Step A: Search for player to get ID (try draft_year - 1)
    search_year = draft_year - 1
    if search_year < 2000: # CFBD data might be sparse before this
        print(f"  Skipping {player_name} as search year {search_year} is too early.")
        continue

    print(f"  Searching for player '{player_name}' from team '{nfl_college_name}' for year {search_year}...")
    try:
        player_search_results = players_api.search_players(
            search_term=player_name,
            team=nfl_college_name, # Be specific with team if possible
            year=search_year      # Look for them in their likely last college year
        )
        time.sleep(0.2) # Basic rate limit

        if player_search_results:
            # Try to find an exact match for name and team
            for p_res in player_search_results:
                # Normalize names for comparison (simple lowercase)
                # API might return slightly different college names, be flexible
                if p_res.name.lower() == player_name.lower() and \
                   (nfl_college_name.lower() in p_res.team_name.lower() if p_res.team_name else False):
                    cfbd_player_id = p_res.id
                    print(f"    Found player ID: {cfbd_player_id} for {p_res.name} at {p_res.team_name}")
                    break
            if not cfbd_player_id and player_search_results: # Fallback: take first result if team matches broadly
                for p_res in player_search_results:
                    if p_res.team_name and nfl_college_name.lower() in p_res.team_name.lower():
                         cfbd_player_id = p_res.id
                         print(f"    Found player ID (fallback by team): {cfbd_player_id} for {p_res.name} at {p_res.team_name}")
                         break
            if not cfbd_player_id:
                 print(f"    Player '{player_name}' not uniquely identified at '{nfl_college_name}' via search for {search_year}.")

    except cfbd.ApiException as e:
        if "401" in str(e):
            print(f"    API Error (401) during player search for {player_name}. Halting API calls.")
            api_key_seems_to_work = False
            continue
        else:
            print(f"    API Error searching for {player_name} ({search_year}): {e}")
            continue
    except Exception as e_gen:
        print(f"    Generic error searching for {player_name} ({search_year}): {e_gen}")
        continue

    if not cfbd_player_id:
        print(f"  No CFBD player ID found for {player_name} from {nfl_college_name} for {search_year}.")
        continue

    # Step B: Get player usage stats for that ID and year
    print(f"  Fetching usage stats for player ID {cfbd_player_id} for year {search_year}...")
    player_usage_stats_for_year = None
    try:
        player_usage_stats_for_year = players_api.get_player_usage(
            year=search_year,
            player_id=cfbd_player_id,
            exclude_garbage_time=False # Set to True if desired
        )
        time.sleep(0.2)
    except cfbd.ApiException as e:
        if "401" in str(e):
            print(f"    API Error (401) fetching usage for {player_name}. Halting API calls.")
            api_key_seems_to_work = False
        else:
            print(f"    API Error fetching usage for {player_name} (ID: {cfbd_player_id}, Year: {search_year}): {e}")
        continue # Skip this player if usage fetch fails
    except Exception as e_gen:
        print(f"    Generic error fetching usage for {player_name} (ID: {cfbd_player_id}, Year: {search_year}): {e_gen}")
        continue

    if player_usage_stats_for_year:
        # player_usage_stats_for_year is a list, usually with one item if player_id is specific
        for usage_stat in player_usage_stats_for_year:
            if usage_stat.id == cfbd_player_id: # Redundant check if API guarantees ID match
                if usage_stat.usage and usage_stat.usage.receiving:
                    # These are attribute names based on cfbd.models.PlayerUsageUsageReceiving
                    recs = usage_stat.usage.receiving.receptions if hasattr(usage_stat.usage.receiving, 'receptions') else 0
                    yds = usage_stat.usage.receiving.yards if hasattr(usage_stat.usage.receiving, 'yards') else 0
                    tds = usage_stat.usage.receiving.tds if hasattr(usage_stat.usage.receiving, 'tds') else 0
                    # Games played is not directly in PlayerUsage. Assuming 1 if stats exist.
                    games = 1 if (recs or yds or tds) else 0

                    if games > 0:
                        player_data_for_year.append({
                            'year': search_year,
                            'player': usage_stat.name or player_name,
                            'college': usage_stat.team_name or nfl_college_name,
                            'games': games,
                            'receptions': recs or 0,
                            'yards': yds or 0,
                            'tds': tds or 0
                        })
                        print(f"      Collected stats for {usage_stat.name} in {search_year}: Rec: {recs}, Yds: {yds}, TDs: {tds}")
                        break # Found stats for this player for this year

    if player_data_for_year:
        prospects_with_some_college_data +=1
        # For this simplified version, career stats are just the single year's stats
        career_summary = {
            'pfr_player_name_nfl': player_name,
            'college_name_nfl': nfl_college_name,
            'cfbd_player_name': player_data_for_year[0]['player'],
            'cfbd_college_name': player_data_for_year[0]['college'],
            'college_games_total': player_data_for_year[0]['games'],
            'college_receptions_total': player_data_for_year[0]['receptions'],
            'college_yards_total': player_data_for_year[0]['yards'],
            'college_tds_total': player_data_for_year[0]['tds'],
            'college_stat_source': 'cfbd_api_single_year'
        }
        college_stats_list.append(career_summary)
    else:
        print(f"  No college stats recorded for {player_name} (ID: {cfbd_player_id}) for year {search_year}.")

print(f"\nProcessed {total_prospects} prospects. Found some college data for {prospects_with_some_college_data} prospects.")
print("-" * 30)

# Step 5: Create a DataFrame from the collected college stats
print("\nStep 5: Creating DataFrame from collected college stats...")
college_stats_df = pd.DataFrame(college_stats_list) if college_stats_list else pd.DataFrame()
if not college_stats_df.empty:
    print(f"College Stats DataFrame shape: {college_stats_df.shape}")
    print(college_stats_df.head())
else:
    print("No college stats collected, DataFrame is empty.")
print("-" * 30)

# Step 6: Clean the college stats DataFrame (basic cleaning)
print("\nStep 6: Cleaning college stats DataFrame...")
if not college_stats_df.empty:
    # Example: df['normalized_name'] = df['cfbd_player_name'].str.lower().str.replace('[^\w\s]', '', regex=True)
    print("Skipping advanced normalization for now.")
else:
    print("College stats DataFrame is empty, skipping cleaning.")
print("-" * 30)

# Step 7: Merge college stats with NFL prospect DataFrame
print("\nStep 7: Merging college stats with NFL prospect DataFrame...")
if not college_stats_df.empty:
    merged_final_df = pd.merge(
        nfl_prospects_df,
        college_stats_df,
        left_on=['pfr_player_name', 'college'], # NFL data name and college
        right_on=['pfr_player_name_nfl', 'college_name_nfl'], # CFBD data original name and college
        how='left'
    )
    # Drop redundant columns used for merge key from the right side
    merged_final_df.drop(columns=['pfr_player_name_nfl', 'college_name_nfl'], inplace=True, errors='ignore')

    successfully_merged_count = merged_final_df['college_receptions_total'].notna().sum() # Example column from college data
    print(f"Shape of merged DataFrame: {merged_final_df.shape}")
    print(f"Total NFL prospects: {len(nfl_prospects_df)}")
    print(f"Prospects for whom college stats were found: {len(college_stats_df)}")
    print(f"Prospects successfully merged with new college stats: {successfully_merged_count}")
else:
    print("College stats DataFrame is empty. Merge skipped. Using original NFL data.")
    merged_final_df = nfl_prospects_df.copy()
    # Add empty columns for college stats to maintain schema if needed for later steps
    # This is important if subsequent steps expect these columns.
    # For now, this is not strictly necessary by the prompt but good practice.
    # cols_to_add = ['cfbd_player_name', 'cfbd_college_name', 'college_games_total',
    #                'college_receptions_total', 'college_yards_total', 'college_tds_total', 'college_stat_source']
    # for col in cols_to_add:
    #    if col not in merged_final_df.columns:
    #        merged_final_df[col] = pd.NA
print("-" * 30)

# Step 8: Log examples of failed merges
print("\nStep 8: Logging failed merges (conceptual)...")
if not college_stats_df.empty and 'successfully_merged_count' in locals():
    if successfully_merged_count < len(college_stats_df): # If some found stats didn't merge
        print(f"Note: {len(college_stats_df) - successfully_merged_count} players had college stats found but didn't merge perfectly (e.g. due to name/college discrepancies if merge keys were different).")
    # Log prospects in NFL list that don't have any college data after merge
    unmerged_nfl_prospects = merged_final_df[merged_final_df['college_stat_source'].isnull()]
    print(f"{len(unmerged_nfl_prospects)} NFL prospects have no college stats after merge attempt.")
    if not unmerged_nfl_prospects.empty:
        print("Examples of NFL prospects with no college stats merged:")
        print(unmerged_nfl_prospects[['pfr_player_name', 'college']].head())
else:
    print("No college stats were available to merge.")
print("-" * 30)

# Step 9: Select relevant columns
print("\nStep 9: Selecting relevant columns (no changes for now)...")
print(f"Current columns: {merged_final_df.columns.tolist()}")
print("-" * 30)

# Step 10: Save the merged DataFrame
print("\nStep 10: Saving the merged DataFrame...")
output_filename = "wr_prospect_data_college.parquet"
try:
    merged_final_df.to_parquet(output_filename, index=False)
    print(f"DataFrame saved successfully to {output_filename}")
    if not api_key_seems_to_work:
        print("WARNING: College data fetching was incomplete due to API key issues.")
    if college_stats_df.empty:
        print("WARNING: No college stats were fetched or merged. The saved file is effectively the same as the input NFL data.")
except Exception as e:
    print(f"Error saving DataFrame to parquet: {e}")
print("-" * 30)

# Step 11: Print .info() summary and .head()
print("\nStep 11: Final DataFrame summary...")
print("\nDataFrame .info():")
merged_final_df.info(verbose=True, show_counts=True) # verbose for full column list
print("\nDataFrame .head():")
print(merged_final_df.head())
print("-" * 30)

if not api_key_seems_to_work:
    print("\nIMPORTANT: CFBD API calls failed due to missing or invalid API key. College stats are likely ABSENT or INCOMPLETE.")

print("College data processing script finished.")
