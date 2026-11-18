import networkx as nx


def analyze_connectivity(graph: nx.MultiDiGraph) -> dict:
    """Return connectivity statistics for a directed road network."""

    if graph.number_of_nodes() == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "weak_components": 0,
            "strong_components": 0,
            "isolated_nodes": 0,
            "largest_weak_component": 0,
        }

    weak_components = list(
        nx.weakly_connected_components(graph)
    )

    strong_components = list(
        nx.strongly_connected_components(graph)
    )

    isolated_nodes = list(nx.isolates(graph))

    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "weak_components": len(weak_components),
        "strong_components": len(strong_components),
        "isolated_nodes": len(isolated_nodes),
        "largest_weak_component": max(
            len(component) for component in weak_components
        ),
    }


def check_transition(graph, from_node, to_node) -> bool:
    """Check if a directed edge connects the given nodes."""

    return graph.has_edge(from_node, to_node)
