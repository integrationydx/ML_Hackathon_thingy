# PathSnap: GPS Map-Matching and Route Reconstruction

**PES2UG24CS052** • **PES2UG24CS55** • **PES2UG24CS010** • **PES2UG24CS015**

PathSnap converts noisy GPS observations into candidate road arcs, predicts the most likely road for each observation, and reconstructs a coherent route with HMM/Viterbi-style decoding utilities. The project includes data preparation, road-network indexing, model benchmarking, Folium maps, a FastAPI backend, and a browser evaluator UI.

## Project overview

The processing pipeline is:

```text
Raw GPS trajectories
        |
        v
Strict parsing and validation
        |
        v
Local projection and candidate road generation
        |
        v
Spatial/kinematic feature engineering
        |
        v
ML candidate classification
        |
        v
Route reconstruction and evaluation
        |
        v
FastAPI backend + evaluator UI
```

## Repository layout

| Directory | Purpose |
|---|---|
| `src/data/` | Raw trip loaders, strict validation, alignment, and EDA |
| `src/features/` | Candidate generation and feature preparation |
| `src/road_network/` | Spatial road indexing, graph construction, and connectivity |
| `src/models/` | Five-model benchmark and model serialization |
| `src/decoder/` | HMM probability preparation, transitions, Viterbi, and route evaluation |
| `api/` | FastAPI application and model-loading logic |
| `ui/` | Browser evaluator interface with Leaflet/Folium-style route display |
| `notebooks/` | EDA notebook |
| `reports/member1/` | Dataset inventory, quality analysis, plots, and Folium HTML |
| `reports/member2/` | Model comparison tables, accuracy log, and plots |
| `tests/` | Pipeline, graph, candidate, HMM, Viterbi, route, and API tests |
| `artifacts/` | Saved champion model and metadata |

## Dataset and preprocessing

The raw dataset is distributed separately and must not contain evaluator data in this public repository.

The pipeline:

- Parses `.track`, `.nodes`, `.arcs`, and optional `.route` files.
- Validates timestamps, coordinates, finite values, and arc endpoints.
- Uses a local metric projection per trip.
- Searches a 50 m candidate radius and retains at most eight road arcs.
- Splits feature segments after timestamp gaps longer than 60 seconds.
- Creates candidate-level features:
  - perpendicular distance
  - heading difference
  - projection ratio
  - candidate distance rank
  - segment length
  - vehicle speed
  - elapsed time
  - heading validity
  - gap indicator
  - zero-length arc indicator
- Keeps reference-derived labels separate from inference input.

Place the external dataset under `data/raw/` and run:

```bash
python3 -m src.features.build_features \
  --data-dir "/absolute/path/to/Student/train" \
  --output-dir data/processed/train \
  --include-labels

python3 -m src.features.build_features \
  --data-dir "/absolute/path/to/Student/student_test" \
  --output-dir data/processed/student_test \
  --include-labels
```

The development alignment is heuristic and is not equivalent to organiser-approved ground truth.

## Model experimentation

The benchmark evaluates five algorithm families:

1. Logistic Regression
2. Distance-Weighted k-NN
3. Random Forest
4. HistGradientBoosting
5. Multi-Layer Perceptron

Run the benchmark when processed data is available:

```bash
python3 -m src.models.benchmark_models
```

The outputs are written to:

- `reports/member2/model_comparison.csv`
- `reports/member2/model_comparison.md`
- `reports/member2/model_accuracy_log.json`
- `reports/member2/model_comparison_plots.png`
- `artifacts/model_final.joblib`
- `artifacts/model_metadata.json`

The current logged MLP result is:

| Metric | Value |
|---|---:|
| CV Macro F1 | 0.8553 |
| Student-Test Top-1 Accuracy | 0.8911 |
| Student-Test Macro F1 | 0.9273 |
| Student-Test Log Loss | 0.1135 |

These metrics reflect the available experiment definition and candidate/derived-label partition. They must not be interpreted as an official result until reproduced against the organiser-approved student-test partition and metric definition.

## HMM and Viterbi route reconstruction

The decoder modules provide:

- `prepare_hmm_inputs()` for emission and transition validation
- `calculate_transition_probability()` for directed-road connectivity and movement consistency
- `viterbi_decode()` for maximum-probability route selection
- `evaluate_route()` for route set precision, recall, F1, and sequence similarity

Run decoder tests:

```bash
python3 -m pytest tests/test_hmm.py tests/test_viterbi.py \
  tests/test_transition_model.py tests/test_route_evaluation.py -q
```

Empty candidate observations, timestamp gaps, disconnected road segments, and zero-probability transitions are handled explicitly rather than silently filled.

## Geographical visualisation

`src/data/eda_inspector.py` generates representative GPS, road-network, and reference-route layers using Folium.

An example interactive map is committed at:

- `reports/member1/trip_00000001.html`

Static EDA visualisations include:

- `reports/member1/dataset_summary.png`
- `reports/member1/representative_trip.png`
- `reports/member2/model_comparison_plots.png`

The evaluator UI overlays GPS observations, reference geometry, and predicted road geometry on Leaflet.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-data.txt
python3 -m pytest -q
```

## Run the API and evaluator

Start the FastAPI server:

```bash
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/
```

Use **Load sample**, edit the candidate JSON if required, and select **Run prediction**.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Browser evaluator UI |
| `GET` | `/health` | Service and model status |
| `GET` | `/metadata` | Champion model and required features |
| `GET` | `/model-comparison` | Logged model benchmark results |
| `POST` | `/predict` | Score grouped road candidates and return a predicted route |

Check the server:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-comparison
```

## Prediction request

Send candidate features generated by the same feature pipeline used for the trained model:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "observations": [{
      "obs_idx": 0,
      "gps": [51.0501, 13.7372],
      "candidates": [{
        "arc_id": 101,
        "features": {
          "distance_m": 4,
          "heading_delta_deg": 8,
          "projection_ratio": 0.4,
          "distance_rank": 1,
          "segment_length_m": 80,
          "speed_mps": 8,
          "dt_s": 1,
          "heading_valid": true,
          "gap_before": false,
          "zero_length_arc": false
        },
        "geometry": [[51.0500, 13.7370], [51.0502, 13.7375]]
      }]
    }],
    "reference_route": [101]
  }'
```

The API returns the selected arc, probability, predicted route, GPS coordinates, and optional reference/predicted geometry.

## Two-laptop LAN deployment

On the server laptop:

```bash
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Find its IP address:

```bash
ipconfig getifaddr en0
```

On the evaluator laptop, open:

```text
http://SERVER_LAPTOP_IP:8000/
```

Test from the evaluator laptop:

```bash
curl http://SERVER_LAPTOP_IP:8000/health
```

Both laptops must be on the same network, and the server laptop firewall must allow inbound TCP connections on port 8000.

## Model artifact and environment

The default champion model is:

- `artifacts/model_final.joblib`
- `artifacts/model_metadata.json`

Override paths for deployment:

```bash
export PATHSNAP_MODEL=/absolute/path/model.joblib
export PATHSNAP_METADATA=/absolute/path/model_metadata.json
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Data and security

- Do not commit raw GPS datasets unless redistribution is authorised.
- Do not commit API keys, passwords, `.env` files, or private credentials.
- `data/`, `*.track`, `*.nodes`, `*.arcs`, `*.route`, and Python caches are ignored.
- Reference routes are development information and should not be used as evaluator input.

## Validation and limitations

- Tests validate parsing, candidate mapping, graph construction, transition probabilities, HMM/Viterbi decoding, route evaluation, and API behavior.
- Missing/large-gap observations and disconnected road networks require explicit handling.
- HMM/Viterbi output is only as reliable as the candidate emissions and transition model.
- Reported model metrics should be verified against the submitted code, prepared data, and official partitions before submission.
