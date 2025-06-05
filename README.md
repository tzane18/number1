# Project Title: Predicting NFL Wide Receiver Success Using Pre-Draft Metrics

## 1. Project Goal/Objective

The primary goal of this project is to develop a machine learning model that predicts the future success of NFL Wide Receiver (WR) prospects based *solely* on pre-draft information. Success is quantified by a metric such as Career Approximate Value (AV) over their first few seasons. The project aims to identify key pre-draft indicators (combine performance, college production, draft position, etc.) that correlate with NFL success.

## 2. Data Sources

*   **NFL Data (via `nfl-data-py` library)**:
    *   Draft Picks (2000-2023): `nfl_data_py.import_draft_picks()` - Provided draft position, player identifiers, college, age, and some NFL career summary statistics (which were later removed for pre-draft modeling).
    *   Combine Data (2000-2023): `nfl_data_py.import_combine_data()` - Provided physical measurements (height, weight) and athletic testing results (40-yard dash, vertical jump, etc.).
    *   Player Information: `nfl_data_py.import_players()` - Provided player metadata like birth dates, full names, and potentially more detailed college information.
    *   Target Variable (`target_AV`): Initially derived from `weighted_career_av` or `career_av` from `nfl_data_py.import_draft_picks()`, representing a player's NFL performance. For this project, `target_AV` was used as provided by this source. *Limitation*: The exact calculation method for `target_AV` (e.g., average over N seasons) was not explicitly re-calculated from raw seasonal PFR AV in this project iteration but taken as a given from the `draft_picks` data after initial exploration showed `nfl_data_py.import_seasonal_data()` lacked direct AV columns.

*   **College Football Data (via `sportsdataverse/cfbfastR-data` GitHub repository)**:
    *   Roster Data: Yearly roster files (e.g., `rosters_YYYY.parquet`) for 2014-2022. Used to link players to their college teams.
        *   Source URL pattern: `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/rosters/parquet/rosters_{year}.parquet`
    *   Play-by-Play (PBP) Style Data: Yearly player stats files (e.g., `player_stats_YYYY.parquet`) for 2014-2022, which contained play-level data.
        *   Source URL pattern: `https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/player_stats/parquet/player_stats_{year}.parquet`
    *   **Limitations**:
        *   College data was only successfully downloaded and processed for the 2014-2022 seasons due to availability under the discovered URL patterns. Data for 1999-2013 could not be found with these patterns.
        *   The PBP data required aggregation to create seasonal summaries (receptions, yards, TDs, games played) for each player.
        *   Only the season immediately prior to a player's NFL draft was used for merging college stats. This resulted in college data being merged for 129 out of 778 prospects.

## 3. Methodology/Process

The project followed a structured approach:

1.  **Initial NFL Data Acquisition & Preprocessing (`00_initial_nfl_data_processing.py`)**:
    *   Loaded draft picks, combine data, and player information using `nfl-data-py`.
    *   Merged these datasets based on player identifiers.
    *   Filtered for Wide Receivers (WRs).
    *   Identified `target_AV` (derived from `w_av` in the draft data).
    *   Saved as `wr_prospect_data_raw.parquet`.

2.  **External College Data Integration (`01_college_data_integration.py`)**:
    *   Attempted `cfbd` API (blocked by lack of API key).
    *   Successfully downloaded yearly roster data and play-by-play (PBP) style player data for 2014-2022 from `sportsdataverse/cfbfastR-data`.
    *   Aggregated PBP data to create season-level statistics (receptions, receiving yards, receiving TDs, games played).
    *   Merged aggregated PBP stats with roster data to link players to their college team names. Player ID standardization (`reception_player_id` from PBP and `athlete_id` from roster) was critical for this merge.
    *   Merged these college seasonal stats (for the season `NFL_draft_year - 1`) into the NFL prospect data.
    *   Saved as `wr_data_with_college.parquet`.

3.  **Iterative Feature Engineering**:
    *   **First Pass & Leakage Discovery (Simulated in `02_final_feature_engineering.py`'s drop list, and `03_model_training_evaluation.py` from previous cycle)**: Initial modeling attempts (not explicitly a separate script in the final flow but part of the iterative process) revealed data leakage from NFL career summary statistics present in the raw `draft_picks` data (e.g., `w_av`, `rec_yards`, `games`). This led to unrealistically high model performance.
    *   **Pre-Draft Feature Engineering (`02_final_feature_engineering.py`)**:
        *   Loaded `wr_data_with_college.parquet`.
        *   **Crucially, explicitly removed all identified NFL career summary statistics** and their `_missing` indicators to prevent data leakage.
        *   Created a `college_stats_missing` indicator. Imputed raw college stats (games, rec, yds, TDs) with 0.
        *   Created derived college features: `college_yards_per_reception`, `college_tds_per_reception`, `college_yards_per_game`.
        *   Calculated `age_at_draft` and `bmi` with robust NaN handling and missing indicators.
        *   Handled missing combine data (median imputation + `_missing` indicators).
        *   Frequency encoded college names (`college_freq_encoded`).
        *   Performed final cleanup of identifiers, descriptive text, and other non-pre-draft columns.
        *   Ensured all remaining features in `X` were numeric and fully imputed.
        *   Saved the final pre-draft feature set as `wr_features_predraft_final.parquet`.

4.  **Model Training and Evaluation (`03_model_training_evaluation.py` - final version)**:
    *   Loaded `wr_features_predraft_final.parquet`.
    *   `target_AV` was separated, and rows with missing `target_AV` were dropped (701 samples remained).
    *   Data was split (80% train / 20% test). Features were scaled for Ridge Regression.
    *   Models Trained: Ridge Regression, Random Forest Regressor, XGBoost Regressor (with early stopping).
    *   Evaluated models on the test set (MSE, RMSE, MAE, R2). Feature importances were extracted.

5.  **Hyperparameter Tuning (`04_hyperparameter_tuning_xgb.py` - conceptual name, actual script `hyperparameter_tuning_xgb.py`)**:
    *   Performed hyperparameter tuning for XGBoost using `RandomizedSearchCV` with R2 scoring.
    *   Evaluated the tuned XGBoost model.

## 4. Key Findings/Results

*   **Model Performance (Pre-Draft Features, Post-Tuning for XGBoost)**:
    *   **Ridge Regression**: Best performing model with R-squared ≈ 0.486 on the initial evaluation.
    *   **Tuned XGBoost Regressor**: R-squared ≈ 0.381. The tuning process did not yield a model outperforming the simpler Ridge model on this feature set.
    *   **Random Forest Regressor (Untuned)**: R-squared ≈ 0.327.
    *   The R-squared values suggest a moderate level of predictability for `target_AV` using the curated pre-draft features.

*   **Key Predictors (from feature importances on pre-draft set)**:
    *   **Draft Position (`pick`)**: Consistently the most important feature.
    *   **Draft Season/Year (`season`)**: Important, possibly capturing class strength or AV scale changes.
    *   **Combine Metrics & Physical Attributes**: `broad_jump`, `cone`, `wt` (combine weight), `forty`, `vertical`, `height`, `bmi` showed relevance.
    *   **Missing Value Indicators**: For XGBoost, indicators like `cone_missing`, `draft_number_missing` were significant, showing that the absence of data can be predictive.
    *   **Age at Draft (`age_at_draft`)**: Relevant.
    *   **College Statistics**: Showed lower importance in top features, likely due to data availability (2014+ seasons, only one year used) and the nature of PBP aggregation. `college_yards_per_game` appeared in XGBoost's top 20.

*   **Challenges & Learnings**:
    *   **Data Leakage**: A major challenge was identifying and removing post-draft NFL career statistics from the feature set to build a true pre-draft predictive model.
    *   **External Data Integration**: Sourcing, downloading, and processing the college data (PBP to seasonal summary, roster linking for team names) was complex and required iterative refinement. Data availability limited the historical depth of college stats.
    *   **Feature Engineering**: Creating meaningful features from available data (e.g., derived college stats, missing indicators) was key.

## 5. Conceptual Code Structure (Simulated in Agent Environment)

While developed in an agent-based environment without explicit file creation until the end, a typical project structure would be:

```
nfl_wr_success_prediction/
|
├── data/                             # For storing downloaded and processed data
│   ├── raw/
│   │   └── wr_prospect_data_raw.parquet  # Initial NFL data merged
│   ├── processed/
│   │   ├── wr_data_with_college.parquet # NFL data + merged college data (PBP based)
│   │   └── wr_features_predraft_final.parquet # Final feature set for modeling
│   └── external_downloads/           # If raw college data were saved locally
│       ├── pbp_data/
│       └── roster_data/
|
├── scripts/                          # Python scripts for each processing stage
│   ├── 00_initial_nfl_data_load.py   # (Simulated by Subtask 1 logic)
│   ├── 01_college_data_integration.py # (Corresponds to Subtask "Download college football player statistics...")
│   ├── 02_feature_engineering_predraft.py # (Corresponds to Subtask "Create a feature set using ONLY pre-draft...")
│   ├── 03_train_evaluate_models.py    # (Corresponds to Subtasks "Train ... final pre-draft" & "Evaluate ... final pre-draft")
│   └── 04_hyperparameter_tuning.py    # (Corresponds to Subtask "Perform hyperparameter tuning...")
|
├── results/                          # For storing model outputs, figures, metrics
│   └── model_evaluation_summary.txt
│   └── feature_importances.csv
|
├── .gitignore
└── README.md
```
*(Note: Script names like `00_...py` are conceptual representations of the agent's subtask execution flow.)*

## 6. Instructions on How to Run (Conceptual)

1.  **Setup**:
    *   Clone the repository (if this were a standard Git project).
    *   Install dependencies: `pip install pandas numpy scikit-learn xgboost nfl-data-py requests pyarrow fastparquet`
      (A `requirements.txt` would normally be provided).

2.  **Data Acquisition**:
    *   The NFL data is fetched via `nfl-data-py`. No manual download needed.
    *   College data is downloaded from `sportsdataverse/cfbfastR-data` by the `01_college_data_integration.py` script.

3.  **Running Scripts**: Execute the Python scripts in the `scripts/` directory in numerical order:
    *   `python scripts/00_initial_nfl_data_load.py` (to generate `wr_prospect_data_raw.parquet`)
    *   `python scripts/01_college_data_integration.py` (to generate `wr_data_with_college.parquet`)
    *   `python scripts/02_feature_engineering_predraft.py` (to generate `wr_features_predraft_final.parquet`)
    *   `python scripts/03_train_evaluate_models.py` (to train and evaluate initial models with pre-draft features)
    *   `python scripts/04_hyperparameter_tuning.py` (to tune XGBoost and evaluate)

## 7. Potential Future Work

*   **Expand College Data**:
    *   Source college data for years prior to 2014 if available from other archives.
    *   Aggregate multiple college seasons (e.g., career totals, breakout year stats) rather than just the year prior to draft.
    *   Incorporate more nuanced college metrics like target share, yards per route run, drop rates, etc., if such data can be sourced or reliably calculated from PBP.
*   **Feature Engineering**:
    *   Explore interaction terms between combine metrics and college production.
    *   Use more sophisticated methods for handling missing combine data (e.g., KNN imputation).
    *   Investigate different encoding schemes for categorical data (e.g., target encoding for `college_name`, carefully managed to prevent leakage).
*   **Modeling**:
    *   Conduct more extensive hyperparameter tuning for all models (Random Forest, Ridge) using `GridSearchCV` or `RandomizedSearchCV`.
    *   Explore other regression algorithms (e.g., SVR, LightGBM, CatBoost, Neural Networks).
    *   Implement ensemble techniques like stacking or voting regressors.
*   **Target Variable Refinement**:
    *   The current `target_AV` is based on `w_av` from `draft_picks`. A more robust target would be to calculate average AV over a fixed period (e.g., first 3 or 5 NFL seasons) using raw seasonal NFL performance data to ensure consistency and avoid potential biases in the pre-calculated `w_av`.
*   **Positional Adjustments**: Consider if adjustments or interactions are needed for players who might have played multiple positions or whose college position differs from their expected NFL role (though the project focused on WRs).

```
