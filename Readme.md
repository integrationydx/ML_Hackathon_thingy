
# PathSnap - Map-Matching & Route Reconstruction (Asphalt-Anarchy)

Noisy GPS points -> candidate road arcs -> ML classifier -> HMM/Viterbi -> coherent route.
FastAPI backend (Laptop 1) + evaluator UI (Laptop 2).

## Team
| Member | Area | Branch |
|---|---|---|
| M1 | Data, EDA, features | feat/data-features |
| M2 | Models, CV, metrics | feat/models-training |
| M3 | Bayesian opt + HMM/Viterbi | feat/tuning-hmm |
| M4 | API, UI, Folium, deployment | feat/api-ui |

## Setup
    python -m venv .venv && source .venv/bin/activate
    python3 -m pip install -r requirements-data.txt

Data goes in data/raw/ (not committed).

## API and evaluator UI

Install the dependencies above, then start the backend from this directory:

    python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000

The API loads `artifacts/model_final.joblib` by default. Override the artifact paths
with `PATHSNAP_MODEL` and `PATHSNAP_METADATA` when deploying another model.

- `GET /health` checks service and model availability.
- `GET /metadata` returns the model name and required feature columns.
- `POST /predict` accepts grouped candidate features and returns the selected arc
  for every observation. See the sample payload in `ui/index.html`.

Open `http://localhost:8000/` in a browser, set the API URL, and click **Run prediction**.
The same evaluator is also available directly as `ui/index.html`.
The UI displays the predicted route and overlays GPS points and predicted candidate
geometry on a Leaflet map. Candidate geometry must be supplied as `[latitude,
longitude]` pairs. The API does not read raw evaluator files or expose reference
labels; reference routes are accepted only for local visual comparison.

## Two-laptop LAN evaluation

Run the API on Laptop 1 (the machine containing the model artifact):

    cd /absolute/path/to/ml-main
    python3 -m pip install -r requirements-data.txt
    python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000

Find Laptop 1's LAN address:

    ipconfig getifaddr en0

If that returns nothing, try:

    ipconfig getifaddr en1

On Laptop 2, while both laptops are on the same Wi-Fi/LAN, open:

    http://LAPTOP_1_IP:8000/

For example, if Laptop 1 reports `192.168.1.25`, use
`http://192.168.1.25:8000/`. The evaluator UI automatically uses the server
address from which it was loaded, so its API URL should remain
`http://192.168.1.25:8000`.

Verify connectivity from Laptop 2 before opening the UI:

    curl http://LAPTOP_1_IP:8000/health

The response should contain `"status":"ok"`. If it cannot connect, allow
incoming connections for Python/port 8000 in Laptop 1's firewall and confirm
both laptops are on the same network. Do not use `127.0.0.1` or `localhost` on
Laptop 2; those addresses refer to Laptop 2 itself.

## Model accuracy log

The five-model results are recorded in
`reports/member2/model_accuracy_log.json` and
`reports/member2/model_comparison.csv`. Re-running
`src/models/benchmark_models.py` regenerates both files from the actual
training and student-test predictions.
