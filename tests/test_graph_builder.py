
import pandas as pd

from src.road_network.graph_builder import build_trip_graph


def test_build_trip_graph():
    roads = pd.DataFrame({
        "trip_id": ["00000001", "00000001"],
        "arc_id": [0, 1],
        "from_node_id": [0, 1],
        "to_node_id": [1, 2],
        "from_lon": [8.5, 8.6],
        "from_lat": [49.8, 49.9],
        "to_lon": [8.6, 8.7],
        "to_lat": [49.9, 50.0],
        "length_m": [100.0, 120.0],
        "heading_deg": [45.0, 50.0],
    })

    graph = build_trip_graph(roads, "00000001")

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph.has_edge(0, 1, key=0)
    assert graph.has_edge(1, 2, key=1)
