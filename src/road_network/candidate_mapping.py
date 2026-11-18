import pandas as pd


def validate_candidate_mapping(
    candidates: pd.DataFrame,
    road_segments: pd.DataFrame
) -> pd.DataFrame:
    """
    Check whether each candidate road segment is present
    in the road network for its corresponding trip.
    """

    required_candidates = {"trip_id", "arc_id"}
    required_roads = {"trip_id", "arc_id"}

    if not required_candidates.issubset(candidates.columns):
        raise ValueError("Candidates must contain trip_id and arc_id")

    if not required_roads.issubset(road_segments.columns):
        raise ValueError("Road segments must contain trip_id and arc_id")

    candidate_data = candidates[["trip_id", "arc_id"]].copy()
    road_data = road_segments[["trip_id", "arc_id"]].copy()

    candidate_data["trip_id"] = candidate_data["trip_id"].astype(str)
    road_data["trip_id"] = road_data["trip_id"].astype(str)

    candidate_data["arc_id"] = pd.to_numeric(
        candidate_data["arc_id"], errors="coerce"
    )
    road_data["arc_id"] = pd.to_numeric(
        road_data["arc_id"], errors="coerce"
    )

    road_data = road_data.dropna().drop_duplicates()

    road_data["arc_id"] = road_data["arc_id"].astype("int64")
    candidate_data["arc_id"] = candidate_data["arc_id"].astype("Int64")

    road_data["exists_in_network"] = True

    result = candidate_data.merge(
        road_data,
        on=["trip_id", "arc_id"],
        how="left"
    )

    result["exists_in_network"] = (
        result["exists_in_network"].fillna(False).astype(bool)
    )

    return result


def calculate_mapping_coverage(mapping: pd.DataFrame) -> dict:
    """Calculate the percentage of candidates found in the road network."""

    total = len(mapping)
    valid = int(mapping["exists_in_network"].sum())
    invalid = total - valid

    return {
        "total_candidates": total,
        "valid_candidates": valid,
        "invalid_candidates": invalid,
        "mapping_coverage_percent": (
            valid / total * 100 if total else 0.0
        ),
    }
