
from pathlib import Path

import pandas as pd
import networkx as nx


REQUIRED_COLUMNS = {
    "trip_id",
    "arc_id",
    "from_node_id",
    "to_node_id",
    "from_lon",
    "from_lat",
    "to_lon",
    "to_lat",
    "length_m",
    "heading_deg",
}


def build_trip_graph(road_segments: pd.DataFrame, trip_id: str) -> nx.MultiDiGraph:
    """
    Build a directed road graph for one trip.

    Each road segment is stored as a directed edge:
        from_node_id -> to_node_id

    The original arc_id is preserved as the MultiDiGraph edge key.
    """
    missing = REQUIRED_COLUMNS - set(road_segments.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    trip_id = str(trip_id)

    trip_roads = road_segments[
        road_segments["trip_id"].astype(str) == trip_id
    ].copy()

    if trip_roads.empty:
        raise ValueError(f"No road segments found for trip_id={trip_id}")

    if trip_roads["arc_id"].duplicated().any():
        raise ValueError(f"Duplicate arc_id found in trip {trip_id}")

    graph = nx.MultiDiGraph(trip_id=trip_id)

    for row in trip_roads.itertuples(index=False):
        graph.add_node(
            int(row.from_node_id),
            longitude=float(row.from_lon),
            latitude=float(row.from_lat),
        )

        graph.add_node(
            int(row.to_node_id),
            longitude=float(row.to_lon),
            latitude=float(row.to_lat),
        )

        graph.add_edge(
            int(row.from_node_id),
            int(row.to_node_id),
            key=int(row.arc_id),
            arc_id=int(row.arc_id),
            length_m=float(row.length_m),
            heading_deg=float(row.heading_deg),
        )

    return graph


def load_road_segments(path: str | Path) -> pd.DataFrame:
    """Load and validate a road_segments Parquet file."""
    road_segments = pd.read_parquet(path)

    missing = REQUIRED_COLUMNS - set(road_segments.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    road_segments["trip_id"] = road_segments["trip_id"].astype(str)

    return road_segments


def build_all_trip_graphs(road_segments: pd.DataFrame) -> dict:
    """Build a separate directed graph for every trip."""
    graphs = {}

    for trip_id, trip_data in road_segments.groupby("trip_id", sort=True):
        graphs[str(trip_id)] = build_trip_graph(trip_data, str(trip_id))

    return graphs
