"""Development-only approximate labels from the ordered reference arc sequence.

This is a monotone spatial alignment, not independently measured point truth.
It never supplies inputs to candidate generation or inference feature extraction.
"""
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

from src.features.spatial import angle_difference


def align_reference(index, observations, *, radius_m=100.0, max_candidates=8,
                    ambiguity_margin_m=3.0, max_label_distance_m=50.0):
    route = index.trip.route
    if route is None:
        raise ValueError('Reference alignment requires an explicitly loaded route')
    geometries = [index.geometries[int(arc)] for arc in route]
    tree = STRtree(geometries)
    options = []
    for row, xy in zip(observations.itertuples(), index.track_xy):
        point = Point(xy)
        positions = tree.query(point, predicate='dwithin', distance=radius_m)
        distances = np.array([point.distance(geometries[p]) for p in positions])
        angles = angle_difference(index.headings[route[positions]], row.heading_deg) if row.heading_valid else np.zeros(len(positions))
        costs = distances + 0.05 * angles
        keep = np.lexsort((positions, costs))[:max_candidates]
        options.append([(int(positions[k]), float(distances[k]), float(costs[k])) for k in keep])
    assigned = [None] * len(observations)
    restart = [False] * len(observations)
    # Backtrack each uninterrupted feasible monotone chain. Missing reference
    # candidates and time gaps split chains instead of forcing invented labels.
    chain = []
    previous = None
    previous_segment = None

    def flush():
        if not chain:
            return
        state = int(np.argmin(chain[-1][2]))
        for obs_idx, choices, costs, parents in reversed(chain):
            assigned[obs_idx] = choices[state]
            state = int(parents[state])
        chain.clear()

    for i, choices in enumerate(options):
        segment = int(observations.iloc[i].segment_id)
        if not choices or segment != previous_segment:
            flush()
            previous = None
        if not choices:
            previous_segment = segment
            continue
        unary = np.array([c[2] for c in choices])
        parents = np.full(len(choices), -1, dtype=int)
        if previous is not None:
            prev_choices, prev_costs = previous
            matrix = np.full((len(prev_choices), len(choices)), np.inf)
            for a, prev_choice in enumerate(prev_choices):
                for b, choice in enumerate(choices):
                    progress = choice[0] - prev_choice[0]
                    if progress >= 0:
                        matrix[a, b] = prev_costs[a] + 0.01 * progress
            parents = np.argmin(matrix, axis=0)
            costs = matrix[parents, np.arange(len(choices))] + unary
            if not np.isfinite(costs).any():
                flush()
                restart[i] = True
                parents[:] = -1
                costs = unary
        else:
            costs = unary
        costs = costs - np.min(costs)
        chain.append((i, choices, costs, parents))
        previous = (choices, costs)
        previous_segment = segment
    flush()
    rows = []
    for i, chosen in enumerate(assigned):
        base = dict(trip_id=index.trip.trip_id, obs_idx=i, alignment_restart=restart[i],
                    label_method='monotone_reference_v1')
        if chosen is None:
            rows.append(dict(base, reference_arc_id=None, route_position=None,
                             alignment_distance_m=None, alignment_margin_m=None,
                             alignment_status='unmatched'))
            continue
        position, distance, cost = chosen
        arc = int(route[position])
        alternatives = [c[2] for c in options[i] if int(route[c[0]]) != arc]
        margin = min(alternatives) - cost if alternatives else None
        ambiguous = margin is not None and margin <= ambiguity_margin_m
        status = 'distant' if distance > max_label_distance_m else ('ambiguous' if ambiguous else 'aligned')
        if restart[i]:
            status = 'restart'
        rows.append(dict(base, reference_arc_id=arc, route_position=position,
                         alignment_distance_m=distance, alignment_margin_m=margin,
                         alignment_status=status))
    result = pd.DataFrame(rows)
    for col in ['reference_arc_id', 'route_position']:
        result[col] = result[col].astype('Int64')
    return result
