
import networkx as nx

from src.decoder.transition_model import (
    calculate_transition_probability,
)


def test_connected_road_transition():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        0, 1, key=10,
        arc_id=10, length_m=100.0
    )

    graph.add_edge(
        1, 2, key=20,
        arc_id=20, length_m=100.0
    )

    probability = calculate_transition_probability(
        graph,
        previous_arc_id=10,
        current_arc_id=20,
        observed_distance_m=200.0,
    )

    assert probability > 0
    assert probability <= 1


def test_disconnected_road_transition():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        0, 1, key=10,
        arc_id=10, length_m=100.0
    )

    graph.add_edge(
        5, 6, key=20,
        arc_id=20, length_m=100.0
    )

    probability = calculate_transition_probability(
        graph,
        previous_arc_id=10,
        current_arc_id=20,
        observed_distance_m=200.0,
    )

    assert probability == 0.0
