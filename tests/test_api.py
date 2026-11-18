import importlib
import json

import numpy as np
from fastapi.testclient import TestClient


def test_health_and_prediction(monkeypatch, tmp_path):
    class FakeModel:
        def predict_proba(self, matrix):
            assert matrix.shape == (2, 10)
            return np.column_stack((1 - matrix[:, 0] / 10, matrix[:, 0] / 10))

    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"features": ["distance_m"] + [
        "heading_delta_deg", "projection_ratio", "distance_rank",
        "segment_length_m", "speed_mps", "dt_s", "heading_valid",
        "gap_before", "zero_length_arc",
    ]}))
    monkeypatch.setattr("joblib.load", lambda _: FakeModel())
    module = importlib.import_module("api.app")
    module.model = FakeModel()
    client = TestClient(module.app)

    assert client.get("/health").status_code == 200
    payload = {
        "observations": [{
            "obs_idx": 0,
            "gps": [51.0, 13.0],
            "candidates": [
                {"arc_id": 1, "features": {name: 1 for name in module.feature_columns},
                 "geometry": [[51.0, 13.0], [51.0, 13.001]]},
                {"arc_id": 2, "features": {name: 2 for name in module.feature_columns}},
            ],
        }],
        "reference_route": [1],
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_route"] == [2]
    assert response.json()["reference_geometry"] == [[[51.0, 13.0], [51.0, 13.001]]]
