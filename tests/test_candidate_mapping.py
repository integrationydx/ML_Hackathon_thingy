
import pandas as pd

from src.road_network.candidate_mapping import (
    validate_candidate_mapping,
    calculate_mapping_coverage,
)


def test_candidate_mapping():
    candidates = pd.DataFrame({
        "trip_id": ["001", "001", "001"],
        "arc_id": [10, 20, 30],
    })

    roads = pd.DataFrame({
        "trip_id": ["001", "001"],
        "arc_id": [10, 20],
    })

    result = validate_candidate_mapping(candidates, roads)

    assert result["exists_in_network"].tolist() == [
        True, True, False
    ]

    metrics = calculate_mapping_coverage(result)

    assert metrics["total_candidates"] == 3
    assert metrics["valid_candidates"] == 2
    assert metrics["invalid_candidates"] == 1
    assert abs(metrics["mapping_coverage_percent"] - (200 / 3)) < 1e-9
