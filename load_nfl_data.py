import nfl_data_py as nfl

# Define the range of years
years = range(2000, 2024)

# Load draft picks
draft_picks = nfl.import_draft_picks(years=years)
print("Draft Picks Columns:")
print(draft_picks.columns)
print("-" * 30)

# Load combine data
# combine_data = nfl.import_combine_data(years=years) # Corrected, but will run one by one.
combine_data = nfl.import_combine_data() # The function doesn't take years.
print("Combine Data Columns:")
print(combine_data.columns)
print("-" * 30)

# Load player data
players = nfl.import_players()
print("Players Columns:")
print(players.columns)
print("-" * 30)

# Load seasonal stats - focusing on 'career_av' or 'weighted_career_av'
# Let's try import_seasonal_data first, then explore others if needed.
seasonal_data = nfl.import_seasonal_data(years=years)
print("Seasonal Data Columns:")
print(seasonal_data.columns)
# Check for AV related columns
av_cols = [col for col in seasonal_data.columns if 'av' in col.lower()]
print(f"AV related columns in seasonal_data: {av_cols}")
print("-" * 30)

# As an alternative for AV, let's check pfr stats if available and more direct
try:
    pfr_seasonal_data = nfl.import_pfr_seasonal_data(years=years)
    print("PFR Seasonal Data Columns:")
    print(pfr_seasonal_data.columns)
    pfr_av_cols = [col for col in pfr_seasonal_data.columns if 'av' in col.lower()]
    print(f"AV related columns in pfr_seasonal_data: {pfr_av_cols}")
except Exception as e:
    print(f"Could not load PFR seasonal data: {e}")
print("-" * 30)

# Load rosters to get more player details if `load_players` is not sufficient
roster_data = nfl.import_rosters(years=years)
print("Roster Data Columns:")
print(roster_data.columns)
print("-" * 30)

print("Data loading script finished.")
