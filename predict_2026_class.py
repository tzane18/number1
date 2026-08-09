"""
Predict NFL success for 2026 WR draft class
This script processes 2026 prospects and generates success predictions
"""
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime

print("="*70)
print("2026 NFL WR Draft Class Success Prediction")
print("="*70)

# Load trained models
print("\n[1/7] Loading trained models...")
with open('ridge_model.pkl', 'rb') as f:
    ridge_model = pickle.load(f)
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
print(f"   ✓ Loaded all models and scaler")
print(f"   ✓ Expected features: {len(feature_columns)}")

# Download 2025 college stats
print("\n[2/7] Downloading 2025 college football stats...")
year = 2025
roster_url = f"https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/rosters/parquet/rosters_{year}.parquet"
stats_url = f"https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/player_stats/parquet/player_stats_{year}.parquet"

try:
    # Download roster data
    print(f"   Downloading roster data for {year}...")
    roster_response = requests.get(roster_url, timeout=30)
    with open(f'rosters_{year}.parquet', 'wb') as f:
        f.write(roster_response.content)
    roster_df = pd.read_parquet(f'rosters_{year}.parquet')
    print(f"   ✓ Loaded {len(roster_df)} roster records")

    # Download stats data
    print(f"   Downloading player stats for {year}...")
    stats_response = requests.get(stats_url, timeout=30)
    with open(f'player_stats_{year}.parquet', 'wb') as f:
        f.write(stats_response.content)
    stats_df = pd.read_parquet(f'player_stats_{year}.parquet')
    print(f"   ✓ Loaded {len(stats_df)} stat records")

except Exception as e:
    print(f"   ✗ Could not download 2025 college data: {e}")
    print(f"   Note: 2025 season data may not be available yet")
    print(f"   Continuing with prospect template...")
    roster_df = None
    stats_df = None

# Process college stats
print("\n[3/7] Processing college statistics...")
if stats_df is not None and roster_df is not None:
    # Filter for plays with receptions (reception_player_id is not null)
    receiving_stats = stats_df[stats_df['reception_player_id'].notna()].copy()

    # Count receptions per player
    rec_counts = receiving_stats.groupby('reception_player_id').size().reset_index(name='college_receptions')

    # Sum yards per player
    yards = receiving_stats.groupby('reception_player_id')['reception_yds'].sum().reset_index()
    yards.columns = ['reception_player_id', 'college_receiving_yards']
    yards['college_receiving_yards'] = yards['college_receiving_yards'].fillna(0)

    # Count games per player
    games = receiving_stats.groupby('reception_player_id')['game_id'].nunique().reset_index()
    games.columns = ['reception_player_id', 'college_games_played']

    # Merge all together
    player_stats = rec_counts.merge(yards, on='reception_player_id')
    player_stats = player_stats.merge(games, on='reception_player_id')

    # Estimate TDs (1 TD per ~70 yards is typical)
    player_stats['college_receiving_tds'] = (player_stats['college_receiving_yards'] / 70).round()

    # Rename ID column and convert to string for matching
    player_stats = player_stats.rename(columns={'reception_player_id': 'athlete_id'})
    player_stats['athlete_id'] = player_stats['athlete_id'].astype(int).astype(str)

    # Ensure roster athlete_id is string
    roster_df = roster_df.copy()
    roster_df['athlete_id'] = roster_df['athlete_id'].astype(str)

    # Merge with roster for player info
    prospects = roster_df.merge(player_stats, on='athlete_id', how='left')

    # Filter for WR position
    if 'position' in prospects.columns:
        prospects = prospects[prospects['position'].str.upper() == 'WR'].copy()

    # Fill NaN values for players without reception stats
    for col in ['college_receptions', 'college_receiving_yards', 'college_games_played', 'college_receiving_tds']:
        if col in prospects.columns:
            prospects[col] = prospects[col].fillna(0)

    # Filter for actual draft prospects (significant production)
    # Keep only WRs with meaningful stats (20+ receptions or 300+ yards)
    prospects = prospects[
        (prospects['college_receptions'] >= 20) |
        (prospects['college_receiving_yards'] >= 300)
    ].copy()

    # Create player names from first_name and last_name
    if 'first_name' in prospects.columns and 'last_name' in prospects.columns:
        prospects['player_name'] = prospects['first_name'].fillna('') + ' ' + prospects['last_name'].fillna('')
        prospects['player_name'] = prospects['player_name'].str.strip()
    elif 'full_name' in prospects.columns:
        prospects['player_name'] = prospects['full_name']
    elif 'athlete_display_name' in prospects.columns:
        prospects['player_name'] = prospects['athlete_display_name']

    # Get college name
    if 'team' in prospects.columns:
        prospects['college_name'] = prospects['team']
    elif 'school' in prospects.columns:
        prospects['college_name'] = prospects['school']

    print(f"   ✓ Found {len(prospects)} WR prospects with significant stats")
else:
    # Create empty template for manual entry
    prospects = pd.DataFrame()
    print(f"   ✓ Created empty prospect template")

# Add key prospects manually (top projected WRs for 2026)
print("\n[4/7] Adding known 2026 WR prospects...")

# Top projected WRs for 2026 draft (these are examples - update with actual prospects)
manual_prospects = [
    {
        'player_name': 'Tetairoa McMillan',
        'college_name': 'Arizona',
        'height': 75,  # 6'3"
        'weight': 212,
        'college_receptions': 90,  # Estimated from 2025 season
        'college_receiving_yards': 1400,
        'college_receiving_tds': 12,
        'college_games_played': 13,
        'birth_year': 2003,
    },
    {
        'player_name': 'Luther Burden III',
        'college_name': 'Missouri',
        'height': 71,  # 5'11"
        'weight': 205,
        'college_receptions': 75,
        'college_receiving_yards': 1100,
        'college_receiving_tds': 10,
        'college_games_played': 13,
        'birth_year': 2004,
    },
    {
        'player_name': 'Emeka Egbuka',
        'college_name': 'Ohio State',
        'height': 73,  # 6'1"
        'weight': 205,
        'college_receptions': 85,
        'college_receiving_yards': 1200,
        'college_receiving_tds': 11,
        'college_games_played': 14,
        'birth_year': 2003,
    },
    {
        'player_name': 'Isaiah Bond',
        'college_name': 'Texas',
        'height': 72,  # 6'0"
        'weight': 185,
        'college_receptions': 65,
        'college_receiving_yards': 950,
        'college_receiving_tds': 8,
        'college_games_played': 13,
        'birth_year': 2004,
    },
    {
        'player_name': 'Kevin Coleman Jr.',
        'college_name': 'Mississippi State',
        'height': 70,  # 5'10"
        'weight': 175,
        'college_receptions': 70,
        'college_receiving_yards': 1050,
        'college_receiving_tds': 9,
        'college_games_played': 12,
        'birth_year': 2003,
    },
]

manual_df = pd.DataFrame(manual_prospects)
print(f"   ✓ Added {len(manual_df)} manually entered prospects")

# Combine with downloaded data if available
if len(prospects) > 0:
    # Merge manual entries
    prospects = pd.concat([prospects, manual_df], ignore_index=True)
else:
    prospects = manual_df

print(f"   ✓ Total prospects: {len(prospects)}")

# Feature engineering
print("\n[5/7] Engineering features for prospects...")

# Initialize season
prospects['season'] = 2026

# Calculate age at draft (draft typically happens in late April)
if 'birth_year' in prospects.columns:
    prospects['age_at_draft'] = 2026 - prospects['birth_year']
elif 'birth_date' in prospects.columns:
    prospects['birth_date_dt'] = pd.to_datetime(prospects['birth_date'], errors='coerce')
    prospects['age_at_draft'] = 2026 - prospects['birth_date_dt'].dt.year
else:
    prospects['age_at_draft'] = 22  # Average draft age

prospects['age_at_draft_missing'] = 0

# Calculate BMI
if 'height' in prospects.columns and 'weight' in prospects.columns:
    prospects['bmi'] = (prospects['weight'] / (prospects['height']**2)) * 703
    prospects['bmi_missing'] = 0
else:
    prospects['bmi'] = 30.0  # Average
    prospects['bmi_missing'] = 1

# College stats
college_stat_cols = ['college_games_played', 'college_receptions',
                     'college_receiving_yards', 'college_receiving_tds']
for col in college_stat_cols:
    if col not in prospects.columns:
        prospects[col] = 0

prospects['college_stats_missing'] = (prospects['college_games_played'] == 0).astype(int)

# Derived college features
prospects['college_yards_per_reception'] = np.where(
    prospects['college_receptions'] > 0,
    prospects['college_receiving_yards'] / prospects['college_receptions'],
    0
)
prospects['college_tds_per_reception'] = np.where(
    prospects['college_receptions'] > 0,
    prospects['college_receiving_tds'] / prospects['college_receptions'],
    0
)
prospects['college_yards_per_game'] = np.where(
    prospects['college_games_played'] > 0,
    prospects['college_receiving_yards'] / prospects['college_games_played'],
    0
)

# College frequency encoding (use training data distribution)
df_train = pd.read_parquet("wr_features_predraft_final.parquet")
college_col = 'college' if 'college' in df_train.columns else None
if college_col and 'college_name' in prospects.columns:
    college_freq_map = df_train[college_col].value_counts(normalize=True).to_dict()
    prospects['college_freq_encoded'] = prospects['college_name'].map(college_freq_map).fillna(0.01)
else:
    prospects['college_freq_encoded'] = 0.01

# Combine metrics - set to median values (will be filled after combine)
combine_cols = {
    'forty': 4.52, 'vertical': 35.0, 'bench': 15.0,
    'broad_jump': 120.0, 'cone': 7.0, 'shuttle': 4.2, 'wt': 200.0
}

for col, median_val in combine_cols.items():
    prospects[col] = median_val
    prospects[f'{col}_missing'] = 1  # Mark as missing since combine hasn't happened

# Other features
prospects['hof'] = 0
prospects['round'] = 3  # Median round
prospects['pick'] = 90  # Median pick
prospects['draft_number'] = 90
prospects['draftround'] = 3
prospects['age'] = prospects['age_at_draft']

# Missing indicators for other fields
other_missing = ['age_missing', 'def_solo_tackles_missing', 'def_ints_missing',
                'def_sacks_missing', 'draft_number_missing', 'draftround_missing',
                'jersey_number_missing']
for col in other_missing:
    prospects[col] = 1

# Other fields
prospects['def_solo_tackles'] = 3.0
prospects['def_ints'] = 1.0
prospects['def_sacks'] = 1.0
prospects['jersey_number'] = 80.0

print(f"   ✓ Features engineered")

# Create feature matrix matching training data
print("\n[6/7] Preparing prediction features...")
X_pred = pd.DataFrame()
for col in feature_columns:
    if col in prospects.columns:
        X_pred[col] = prospects[col]
    else:
        # Use median from training data
        if col in df_train.columns:
            X_pred[col] = df_train[col].median()
        else:
            X_pred[col] = 0

# Fill any remaining NaN values with median from training data
for col in X_pred.columns:
    if X_pred[col].isnull().any():
        if col in df_train.columns:
            fill_value = df_train[col].median()
        else:
            fill_value = 0
        X_pred[col] = X_pred[col].fillna(fill_value)

# Ensure all columns are numeric
X_pred = X_pred.apply(pd.to_numeric, errors='coerce')

# Final NaN check and fill
X_pred = X_pred.fillna(0)

print(f"   ✓ Feature matrix prepared: {X_pred.shape}")
print(f"   ✓ NaN values: {X_pred.isnull().sum().sum()}")

# Make predictions
print("\n[7/7] Generating predictions...")

# Ridge prediction (best model)
X_pred_scaled = scaler.transform(X_pred)
ridge_preds = ridge_model.predict(X_pred_scaled)

# Other models
rf_preds = rf_model.predict(X_pred)
xgb_preds = xgb_model.predict(X_pred)

# Create results dataframe
results = pd.DataFrame({
    'Player': prospects.get('player_name', prospects.get('full_name', 'Unknown')),
    'College': prospects.get('college_name', prospects.get('college', 'Unknown')),
    'Height': prospects.get('height', 0),
    'Weight': prospects.get('weight', 0),
    'College_Rec': prospects.get('college_receptions', 0),
    'College_Yards': prospects.get('college_receiving_yards', 0),
    'College_TDs': prospects.get('college_receiving_tds', 0),
    'YPG': prospects.get('college_yards_per_game', 0).round(1),
    'Ridge_Pred_AV': ridge_preds.round(1),
    'RF_Pred_AV': rf_preds.round(1),
    'XGB_Pred_AV': xgb_preds.round(1),
    'Avg_Pred_AV': ((ridge_preds + rf_preds + xgb_preds) / 3).round(1),
})

# Add success classification
results['Success_Tier'] = pd.cut(
    results['Avg_Pred_AV'],
    bins=[0, 10, 20, 30, 100],
    labels=['Below Average', 'Average', 'Above Average', 'Elite']
)

# Sort by average prediction
results = results.sort_values('Avg_Pred_AV', ascending=False)

print(f"   ✓ Predictions generated for {len(results)} prospects")

# Display results
print("\n" + "="*70)
print("2026 WR DRAFT CLASS PREDICTIONS")
print("="*70)
print("\nTop 20 Prospects by Predicted NFL Success:")
print(results.head(20).to_string(index=False))

# Save results
results.to_csv('2026_wr_predictions.csv', index=False)
print(f"\n✓ Full results saved to '2026_wr_predictions.csv'")

# Generate summary statistics
print("\n" + "="*70)
print("PREDICTION SUMMARY")
print("="*70)
print(f"\nTotal Prospects Evaluated: {len(results)}")
print(f"\nPredicted Success Distribution:")
print(results['Success_Tier'].value_counts().to_string())
print(f"\nAverage Predicted Career AV: {results['Avg_Pred_AV'].mean():.1f}")
print(f"Median Predicted Career AV: {results['Avg_Pred_AV'].median():.1f}")
print(f"Predicted Elite WRs (AV > 30): {(results['Avg_Pred_AV'] > 30).sum()}")
print(f"Predicted Above Avg WRs (AV > 20): {(results['Avg_Pred_AV'] > 20).sum()}")

print("\n" + "="*70)
print("Note: Predictions use estimated draft positions (Round 3, Pick 90)")
print("Predictions will be more accurate after:")
print("  - NFL Combine (Feb/March 2026)")
print("  - Actual draft selections (April 2026)")
print("="*70)
