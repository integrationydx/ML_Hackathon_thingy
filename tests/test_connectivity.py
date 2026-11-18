
import networkx as nx

from src.road_network.connectivity_check import (
    analyze_connectivity,
    check_transition,
)


def test_analyze_connectivity():
    graph = nx.MultiDiGraph()

    graph.add_edge(0, 1, key=0)
    graph.add_edge(1, 2, key=1)
    graph.add_edge(3, 4, key=2)

    result = analyze_connectivity(graph)

    assert result["num_nodes"] == 5
    assert result["num_edges"] == 3
    assert result["weak_components"] == 2
    assert result["isolated_nodes"] == 0
    assert result["largest_weak_component"] == 3


def test_check_transition():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, key=0)

    assert check_transition(graph, 0, 1) is True
    assert check_transition(graph, 1, 0) is False
