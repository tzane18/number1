"""
Enhanced 2026 WR Draft Class Prediction with Draft Scenarios
This version creates predictions across different draft scenarios
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("2026 NFL WR Draft Class Success Prediction (Enhanced)")
print("="*70)

# Load trained models
print("\n[1/5] Loading trained models...")
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
print(f"   ✓ Loaded Random Forest and XGBoost models")
print(f"   Note: Using RF & XGBoost only (more robust for unknown draft positions)")

# Load prospect data from previous run
print("\n[2/5] Loading processed prospect data...")
prospects_file = '2026_wr_predictions.csv'
try:
    prospects = pd.read_csv(prospects_file)
    # Filter for meaningful prospects (top producers)
    prospects = prospects[(prospects['College_Rec'] >= 40) | (prospects['College_Yards'] >= 600)]
    print(f"   ✓ Loaded {len(prospects)} high-production prospects")
except:
    print("   ✗ Could not load previous results. Run predict_2026_class.py first.")
    exit()

# Re-load original feature data for scenarios
print("\n[3/5] Preparing draft scenarios...")
df_train = pd.read_parquet("wr_features_predraft_final.parquet")

# Load the full prospect dataset with all features
# We'll need to reprocess from the saved parquet files
import requests

year = 2025
try:
    roster_df = pd.read_parquet(f'rosters_{year}.parquet')
    stats_df = pd.read_parquet(f'player_stats_{year}.parquet')

    # Reprocess reception stats
    receiving_stats = stats_df[stats_df['reception_player_id'].notna()].copy()
    rec_counts = receiving_stats.groupby('reception_player_id').size().reset_index(name='college_receptions')
    yards = receiving_stats.groupby('reception_player_id')['reception_yds'].sum().reset_index()
    yards.columns = ['reception_player_id', 'college_receiving_yards']
    yards['college_receiving_yards'] = yards['college_receiving_yards'].fillna(0)
    games = receiving_stats.groupby('reception_player_id')['game_id'].nunique().reset_index()
    games.columns = ['reception_player_id', 'college_games_played']

    player_stats = rec_counts.merge(yards, on='reception_player_id')
    player_stats = player_stats.merge(games, on='reception_player_id')
    player_stats['college_receiving_tds'] = (player_stats['college_receiving_yards'] / 70).round()
    player_stats = player_stats.rename(columns={'reception_player_id': 'athlete_id'})
    player_stats['athlete_id'] = player_stats['athlete_id'].astype(int).astype(str)

    roster_df = roster_df.copy()
    roster_df['athlete_id'] = roster_df['athlete_id'].astype(str)
    prospects_full = roster_df.merge(player_stats, on='athlete_id', how='left')

    # Filter for WRs with production
    if 'position' in prospects_full.columns:
        prospects_full = prospects_full[prospects_full['position'].str.upper() == 'WR'].copy()

    for col in ['college_receptions', 'college_receiving_yards', 'college_games_played', 'college_receiving_tds']:
        if col in prospects_full.columns:
            prospects_full[col] = prospects_full[col].fillna(0)

    prospects_full = prospects_full[
        (prospects_full['college_receptions'] >= 40) |
        (prospects_full['college_receiving_yards'] >= 600)
    ].copy()

    # Create player names
    if 'first_name' in prospects_full.columns and 'last_name' in prospects_full.columns:
        prospects_full['player_name'] = prospects_full['first_name'].fillna('') + ' ' + prospects_full['last_name'].fillna('')
        prospects_full['player_name'] = prospects_full['player_name'].str.strip()

    if 'team' in prospects_full.columns:
        prospects_full['college_name'] = prospects_full['team']

    print(f"   ✓ Loaded {len(prospects_full)} top prospects for scenario analysis")

except Exception as e:
    print(f"   ✗ Error loading prospect data: {e}")
    exit()


def create_features_for_scenario(prospects_df, draft_round, start_pick=None):
    """Create feature matrix for a given draft scenario"""
    prospects_scenario = prospects_df.copy()

    # Set draft scenario
    prospects_scenario['season'] = 2026
    prospects_scenario['round'] = draft_round
    if start_pick is None:
        # Estimate picks based on round
        pick_starts = {1: 1, 2: 33, 3: 65, 4: 105, 5: 145, 6: 185, 7: 225}
        start_pick = pick_starts.get(draft_round, 90)

    prospects_scenario['pick'] = start_pick + np.arange(len(prospects_scenario))
    prospects_scenario['draft_number'] = prospects_scenario['pick']
    prospects_scenario['draftround'] = draft_round

    # Calculate age at draft
    prospects_scenario['age_at_draft'] = 22.0  # Average
    prospects_scenario['age'] = 22.0
    prospects_scenario['age_at_draft_missing'] = 1
    prospects_scenario['age_missing'] = 1

    # Calculate BMI
    if 'height' in prospects_scenario.columns and 'weight' in prospects_scenario.columns:
        prospects_scenario['bmi'] = (prospects_scenario['weight'] / (prospects_scenario['height']**2)) * 703
        prospects_scenario['bmi_missing'] = 0
    else:
        prospects_scenario['bmi'] = 30.0
        prospects_scenario['bmi_missing'] = 1

    # College stats processing
    prospects_scenario['college_stats_missing'] = (prospects_scenario['college_games_played'] == 0).astype(int)

    prospects_scenario['college_yards_per_reception'] = np.where(
        prospects_scenario['college_receptions'] > 0,
        prospects_scenario['college_receiving_yards'] / prospects_scenario['college_receptions'],
        0
    )
    prospects_scenario['college_tds_per_reception'] = np.where(
        prospects_scenario['college_receptions'] > 0,
        prospects_scenario['college_receiving_tds'] / prospects_scenario['college_receptions'],
        0
    )
    prospects_scenario['college_yards_per_game'] = np.where(
        prospects_scenario['college_games_played'] > 0,
        prospects_scenario['college_receiving_yards'] / prospects_scenario['college_games_played'],
        0
    )

    # College encoding
    college_freq_map = df_train['college' if 'college' in df_train.columns else 'season'].value_counts(normalize=True).to_dict()
    if 'college_name' in prospects_scenario.columns:
        prospects_scenario['college_freq_encoded'] = prospects_scenario['college_name'].map(college_freq_map).fillna(0.01)
    else:
        prospects_scenario['college_freq_encoded'] = 0.01

    # Combine metrics
    combine_cols = {
        'forty': 4.52, 'vertical': 35.0, 'bench': 15.0,
        'broad_jump': 120.0, 'cone': 7.0, 'shuttle': 4.2, 'wt': 200.0
    }
    for col, median_val in combine_cols.items():
        prospects_scenario[col] = median_val
        prospects_scenario[f'{col}_missing'] = 1

    # Other fields
    prospects_scenario['hof'] = 0
    prospects_scenario['def_solo_tackles'] = 3.0
    prospects_scenario['def_ints'] = 1.0
    prospects_scenario['def_sacks'] = 1.0
    prospects_scenario['jersey_number'] = 80.0
    prospects_scenario['def_solo_tackles_missing'] = 1
    prospects_scenario['def_ints_missing'] = 1
    prospects_scenario['def_sacks_missing'] = 1
    prospects_scenario['draft_number_missing'] = 0
    prospects_scenario['draftround_missing'] = 0
    prospects_scenario['jersey_number_missing'] = 1

    # Create feature matrix
    X_pred = pd.DataFrame()
    for col in feature_columns:
        if col in prospects_scenario.columns:
            X_pred[col] = prospects_scenario[col]
        else:
            if col in df_train.columns:
                X_pred[col] = df_train[col].median()
            else:
                X_pred[col] = 0

    # Fill NaNs
    for col in X_pred.columns:
        if X_pred[col].isnull().any():
            if col in df_train.columns:
                fill_value = df_train[col].median()
            else:
                fill_value = 0
            X_pred[col] = X_pred[col].fillna(fill_value)

    X_pred = X_pred.apply(pd.to_numeric, errors='coerce')
    X_pred = X_pred.fillna(0)

    return X_pred, prospects_scenario


# Generate predictions for different draft scenarios
print("\n[4/5] Generating predictions across draft scenarios...")

scenarios = {
    'Round 1 (Elite Picks 1-32)': 1,
    'Round 2 (Picks 33-64)': 2,
    'Round 3 (Picks 65-104)': 3,
    'Round 4-5 (Day 3)': 4,
}

all_results = []

for scenario_name, round_num in scenarios.items():
    X_pred, prospects_scenario = create_features_for_scenario(prospects_full, round_num)

    # Make predictions
    rf_preds = rf_model.predict(X_pred)
    xgb_preds = xgb_model.predict(X_pred)
    avg_preds = (rf_preds + xgb_preds) / 2

    # Store results
    scenario_results = pd.DataFrame({
        'Player': prospects_scenario.get('player_name', 'Unknown'),
        'College': prospects_scenario.get('college_name', 'Unknown'),
        'Height': prospects_scenario.get('height', 0),
        'Weight': prospects_scenario.get('weight', 0),
        'College_Rec': prospects_scenario.get('college_receptions', 0),
        'College_Yards': prospects_scenario.get('college_receiving_yards', 0),
        'College_TDs': prospects_scenario.get('college_receiving_tds', 0),
        'YPG': prospects_scenario.get('college_yards_per_game', 0).round(1),
        'Scenario': scenario_name,
        'Draft_Round': round_num,
        'RF_Pred_AV': rf_preds.round(1),
        'XGB_Pred_AV': xgb_preds.round(1),
        'Avg_Pred_AV': avg_preds.round(1),
    })

    all_results.append(scenario_results)

combined_results = pd.concat(all_results, ignore_index=True)

print(f"   ✓ Generated predictions for {len(scenarios)} scenarios")

# Analyze results
print("\n[5/5] Creating summary report...")

# Get Round 1 scenario for top prospects
round1_results = combined_results[combined_results['Draft_Round'] == 1].copy()
round1_results = round1_results.sort_values('Avg_Pred_AV', ascending=False)

# Calculate production metrics for ranking
round1_results['Production_Score'] = (
    round1_results['College_Rec'] * 0.3 +
    round1_results['College_Yards'] * 0.01 +
    round1_results['College_TDs'] * 2.0 +
    round1_results['YPG'] * 0.5
)

round1_results = round1_results.sort_values('Production_Score', ascending=False)

# Display results
print("\n" + "="*70)
print("2026 WR DRAFT CLASS - TOP PROSPECTS")
print("="*70)
print("\nRound 1 Scenario (if drafted in Round 1):")
print(round1_results.head(25)[['Player', 'College', 'College_Rec', 'College_Yards',
                                'YPG', 'Avg_Pred_AV', 'Production_Score']].to_string(index=False))

# Save all scenarios
combined_results.to_csv('2026_wr_predictions_scenarios.csv', index=False)
round1_results.to_csv('2026_wr_top_prospects.csv', index=False)

print(f"\n✓ Saved scenario predictions to '2026_wr_predictions_scenarios.csv'")
print(f"✓ Saved top prospects to '2026_wr_top_prospects.csv'")

# Summary statistics
print("\n" + "="*70)
print("PREDICTION SUMMARY BY DRAFT ROUND")
print("="*70)
for scenario_name, round_num in scenarios.items():
    scenario_data = combined_results[combined_results['Draft_Round'] == round_num]
    print(f"\n{scenario_name}:")
    print(f"  Average Predicted AV: {scenario_data['Avg_Pred_AV'].mean():.1f}")
    print(f"  Predicted Elite (AV > 30): {(scenario_data['Avg_Pred_AV'] > 30).sum()}")
    print(f"  Predicted Above Avg (AV > 20): {(scenario_data['Avg_Pred_AV'] > 20).sum()}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
