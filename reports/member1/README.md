# Member 1 data pipeline handoff

This component implements raw parsing, dataset inventory, spatial candidates, inference features and **approximate** development-reference alignment. It does not implement trained models, HMM route decoding, or the API. Branch: `feature/m1-data-pipeline`.

## Run from the repository root

Python 3.10 or later is required. Keep the original dataset outside the public repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
export PATHSNAP_DATA_ROOT="/absolute/path/to/Student folder"
python -m src.data.eda_inspector --input-dir "$PATHSNAP_DATA_ROOT" --output-dir reports/member1 --map-trip 00000001
python -m src.features.build_features --data-dir "$PATHSNAP_DATA_ROOT/train" --output-dir data/processed/v1/train --include-labels
python -m src.features.build_features --data-dir "$PATHSNAP_DATA_ROOT/student_test" --output-dir data/processed/v1/student_test --include-labels
python -m pytest tests/data -q
```

Use **new output directories** for each preparation run: existing destinations are refused to prevent stale labels or mixed artifacts. A successful output contains `manifest.json`; incomplete staging is not a valid dataset. No reserved evaluator data was supplied or used.

For a quick real-trip run, add `--limit 1` and choose a different output directory. Defaults are a 50m candidate radius, at most eight candidates, a 60s gap threshold and a 100m reference alignment search. These are initial engineering choices, not tuned optima. CLI flags are `--radius-m`, `--max-candidates`, `--gap-s`, `--reference-radius-m`.

## Outputs per partition

| File | Purpose |
|---|---|
| `candidates.parquet` | All inference candidate rows, with no reference or target columns |
| `observations.parquet` | Every GPS observation, including zero-candidate observations, timing and gap flags |
| `road_segments.parquet` | All directed road arcs and WGS84 endpoints for Member 3 |
| `reference_labels.parquet` | Separate development-only labels, uncertainty, candidate-miss and eligibility flags |
| `train_features.parquet` | Convenience candidate/label join restricted to eligible observations; **student-test output must never be used for fitting** |
| `feature_columns.json` | Ordered numeric/boolean model feature allowlist |
| `manifest.json` | Parameters, projection per trip, schema version, trip counts and label-quality diagnostics |

The complete output data stays under ignored `/data/`. Source code, inventory, summaries, plots, map generation and notebooks are trackable. Do not add the raw or derived dataset to the public repository without dataset redistribution permission.

## For Member 2

```python
import json
import pandas as pd
from sklearn.model_selection import GroupKFold

root = 'data/processed/v1/train'
features = json.load(open(root + '/feature_columns.json'))
rows = pd.read_parquet(root + '/train_features.parquet')
X = rows[features]             # ONLY the feature allowlist
y = rows['target']            # candidate correctness, binary
```

Use `groups=rows['trip_id']` with `GroupKFold`. Fit learned preprocessing separately in each training fold. The column allowlist intentionally excludes target, reference arc, route position, alignment distance/margin/status and eligibility: **never train on all numeric columns**.

For evaluation, score the full `candidates.parquet`, select the highest score per `(trip_id, obs_idx)`, and join separate reference labels afterwards. Keep the full `observations.parquet` as the denominator ledger so missing candidate sets are visible. Report label coverage and metrics on clearly defined subsets. Define observation Macro F1 consistently, e.g. compute within each trip's local arc label space then macro-average trips. Candidate binary F1 is a different metric and must be labelled separately. Candidate miss rate here is relative to derived known reference labels, not an official independent ground truth.

## For Member 3

Use `(trip_id, arc_id)` for a road segment, not `arc_id` alone. `from_node_id` and `to_node_id` define direction; reverse arcs remain separate. Use all `road_segments.parquet` rows for intermediate paths, not just GPS candidates. Preserve `(trip_id, obs_idx, candidate_idx)` when joining classifier scores. `projected_x_m` and `projected_y_m` use the per-trip projection in the manifest. `segment_id` increments after a gap over 60s; choose and document decoder behaviour at these boundaries.

## For Member 4

```python
from src.data.loaders import load_trip
from src.features.build_features import prepare_trip, FEATURE_COLUMNS

trip = load_trip('/local/server/path/00000001')  # does not read .route
prepared = prepare_trip(trip)                  # labels remain None
X = prepared.candidates[FEATURE_COLUMNS]
```

Use the model's recorded preparation parameters at inference. Empty candidate sets are valid, explicit outcomes; handle them in the decoder and UI. Load/evaluate references separately on the server only after prediction. The included Folium map is a **development EDA map**, not the reserved evaluation interface; it intentionally has a reference layer.

## Features and units

- `distance_m`: perpendicular distance to the complete segment, including its endpoints.
- `heading_delta_deg`: smallest directed heading difference, 0 to 180; zero with a false `heading_valid` when unavailable.
- `projection_ratio`: projected fraction along the segment, 0 to 1.
- `distance_rank`: 1-based rank; `candidate_idx` is 0-based and stable after distance/arc-ID sorting.
- `segment_length_m`: directed segment geometry length.
- `speed_mps`: previous displacement divided by elapsed time; zero at first observation or after a gap. Consult observations' `speed_valid`; those zeros do not prove a stationary vehicle.
- `dt_s`, `heading_valid`, `gap_before`, `zero_length_arc`: elapsed time and explicit quality indicators.

No scaling or learned imputation is fitted here. GPS rows are preserved. Coordinates are validated but not moved to the reference route. A local azimuthal equidistant projection is centred on each trip, rather than using one UTM zone globally. Local segment distances are an approximation away from the projection centre; validate distortion for unusually broad trips before extending to other datasets.

## Reference alignment limitations

Reference labels are derived separately by searching reference-route segments within 100m, keeping eight candidate **route occurrences**, and performing monotone dynamic programming. Cost is distance plus 0.05m per degree of heading difference and 0.01m per forward route-position step. Matching may remain on a segment or advance; it cannot move backwards inside a chain. Large timestamp gaps, missing reference candidates, and infeasible monotone transitions split chains.

Statuses are `aligned`, `ambiguous`, `distant`, `restart` or `unmatched`. Distance over 50m is distant; a local alternative distinct arc within a 3m unary-cost margin is ambiguous. This margin is a conservative heuristic, not a calibrated confidence probability. Only aligned observations whose derived reference arc is in the inference candidate set enter the convenience training table. Restarts and all other observations remain in separate outputs for audit/evaluation. Unknown reference labels have NA candidate-miss flags, not false successes.

This method can be wrong at loops, parallel roads, intersections, noisy headings and reference discontinuities. It does not enforce speed-consistent route progress and is not the final HMM decoder. Obtain organiser clarification on expected observation-label construction before claiming official observation-level accuracy. A strong score against these labels alone does not prove physical route correctness.

## EDA evidence

`dataset_inventory.csv`, `validation_errors.json`, `dataset_summary.png`, `representative_trip.png`, and `trip_00000001.html` are generated from the actual permitted dataset. Map background roads are capped at 3,000 for display; GPS markers are capped at 1,500. Candidate generation always uses the full road network. Open the HTML map in a browser with network access for map tiles and Leaflet assets.
