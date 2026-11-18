"""Calculate transition scores between connected road segments."""

import math

import networkx as nx


def calculate_transition_probability(
    graph: nx.MultiDiGraph,
    previous_arc_id: int,
    current_arc_id: int,
    observed_distance_m: float,
    beta: float = 50.0,
) -> float:
    """
    Estimate the probability of moving from one road segment to another.

    The score considers directed road connectivity and how closely the
    road-network distance matches the distance observed from GPS movement.
    """

    if beta <= 0:
        raise ValueError("beta must be greater than zero")

    # Find all graph edges that correspond to the two candidate road arcs.
    previous_edges = [
        (u, v, data)
        for u, v, key, data in graph.edges(
            keys=True, data=True
        )
        if data.get("arc_id") == previous_arc_id
    ]

    current_edges = [
        (u, v, data)
        for u, v, key, data in graph.edges(
            keys=True, data=True
        )
        if data.get("arc_id") == current_arc_id
    ]

    # A transition cannot be scored if either road arc is missing.
    if not previous_edges or not current_edges:
        return 0.0

    best_probability = 0.0

    for _, previous_to, previous_data in previous_edges:
        for current_from, _, current_data in current_edges:

            # Measure the shortest directed path between the two road arcs.
            try:
                if previous_to == current_from:
                    network_distance = 0.0
                else:
                    network_distance = nx.shortest_path_length(
                        graph,
                        source=previous_to,
                        target=current_from,
                        weight="length_m",
                    )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            # Include the remaining length of the previous arc, the connecting
            # network path, and the length of the current arc.
            road_distance = (
                float(previous_data["length_m"])
                + float(network_distance)
                + float(current_data["length_m"])
            )

            error = abs(road_distance - observed_distance_m)

            # Smaller differences between road and GPS distances get higher scores.
            probability = math.exp(-error / beta)

            best_probability = max(
                best_probability, probability
            )

    return best_probability
