
from pathlib import Path
import pandas as pd

from src.road_network.graph_builder import load_road_segments
from src.road_network.candidate_mapping import (
    validate_candidate_mapping,
    calculate_mapping_coverage,
)

DATA_DIR = Path("data/processed/v1/train")

print("Loading road segments...")
roads = load_road_segments(DATA_DIR / "road_segments.parquet")

print("Loading candidates...")
candidates = pd.read_parquet(DATA_DIR / "candidates.parquet")

print("\nValidating candidate-to-road mapping...")
mapping = validate_candidate_mapping(candidates, roads)

metrics = calculate_mapping_coverage(mapping)

print("\n===== CANDIDATE MAPPING RESULTS =====")
for key, value in metrics.items():
    print(f"{key}: {value}")

unique_pairs = mapping.drop_duplicates(["trip_id", "arc_id"])

print("\nUnique candidate trip-arc pairs:", len(unique_pairs))
print("Unique road trip-arc pairs:",
      len(roads.drop_duplicates(["trip_id", "arc_id"])))
