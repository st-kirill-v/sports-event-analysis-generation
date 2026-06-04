# Match Dynamics Project Structure

## Active Entry Points

- `scripts/run_football.py`
  Short alias for the active Football pipeline.

- `scripts/run_nba.py`
  Short alias for the active NBA pipeline.

- `scripts/run_pipeline.py`
  Common orchestrator for Football and/or NBA.

- `scripts/run_data_audit_ui.py`
  Starts the Streamlit UI.

## Script Folders

- `scripts/common/`
  UI/audit/common orchestration scripts.

- `scripts/football/`
  Football merge, preprocessing, feature engineering, sequence building,
  LSTM training, ablation, threshold tuning, calibration and error analysis scripts.

- `scripts/nba/`
  NBA merge, preprocessing, feature engineering and LSTM regression scripts.
  The 200-game merge command is `scripts/nba/merge_nba_200.py`.

## Source Packages

The actual implementation is split by namespace:

- `src/match_dynamics/common/`
  - config
  - data loading
  - metrics/evaluation
  - generic models
  - sequence helpers
  - visualization helpers
  - combined orchestration

- `src/match_dynamics/football_pipeline/`
  - football merge
  - event preprocessing
  - feature engineering
  - sequence dataset construction
  - LSTM training
  - feature ablation
  - threshold tuning
  - calibration
  - error analysis

- `src/match_dynamics/nba_pipeline/`
  - NBA movement parsing
  - events/shots/movement merge
  - preprocessing
  - clutch-time feature engineering
  - LSTM regression training

- `src/match_dynamics/ui/`
  - Streamlit UI
  - audit helpers

The top-level files in `src/match_dynamics/*.py` are compatibility wrappers.
They preserve old imports such as `match_dynamics.config` while the real code
lives in the namespace folders above.

## Active UI Tabs

- `Overview`
- `Football Merge`
- `Football Merged Processed`
- `Football Merged Feature Engineering`
- `NBA Merge`
- `NBA Merge Processing`
- `NBA Merge Feature Engineering`
- `Football Metrics`
- `NBA Metric`
- `Conclusion`

Removed from active UI:

- `NBA Raw`
- `NBA Processed`
- `NBA Join Quality`
- `Feature Engineering`
- `Correlations`
- `Model Metrics`

## Archive

Old exploratory or legacy scripts were moved to:

- `archive/scripts_legacy/`

They were not deleted, so they can be restored if needed.
