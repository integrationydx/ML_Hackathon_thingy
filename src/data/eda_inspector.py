"""Dataset inventory and reproducible EDA plots/maps for permitted development data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.loaders import load_trip, trip_directories


def inspect_trip(trip):
    nodes = len(trip.nodes)
    graph = coo_matrix((np.ones(len(trip.arcs)), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(nodes, nodes)).tocsr()
    components = connected_components(graph, directed=True, connection='weak', return_labels=False)
    dt = np.diff(trip.track[:, 2])
    route = trip.route
    return dict(trip_id=trip.trip_id, observations=len(trip.track), nodes=nodes,
        arcs=len(trip.arcs), reference_arcs=len(route) if route is not None else None,
        duration_s=float(trip.track[-1, 2] - trip.track[0, 2]),
        median_dt_s=float(np.median(dt)) if len(dt) else 0,
        max_gap_s=float(dt.max()) if len(dt) else 0,
        steps=len(dt), one_second_steps=int((dt == 1).sum()), gaps_over_60s=int((dt > 60).sum()),
        longitude_min=float(trip.track[:, 0].min()), longitude_max=float(trip.track[:, 0].max()),
        latitude_min=float(trip.track[:, 1].min()), latitude_max=float(trip.track[:, 1].max()),
        weak_components=int(components), isolated_nodes=int(nodes - len(np.unique(trip.arcs))),
        zero_length_arcs=int(np.all(trip.nodes[trip.arcs[:, 0]] == trip.nodes[trip.arcs[:, 1]], axis=1).sum()),
        reference_nonjoining_pairs=int((trip.arcs[route[:-1], 1] != trip.arcs[route[1:], 0]).sum()) if route is not None else None)


def make_map(trip, output, *, max_roads=3000, max_points=1500):
    import folium
    from shapely.geometry import box
    from src.features.spatial import RoadIndex
    index = RoadIndex(trip)
    centre = [float(np.median(trip.track[:, 1])), float(np.median(trip.track[:, 0]))]
    result = folium.Map(location=centre, zoom_start=13)
    minxy = index.track_xy.min(axis=0) - 150
    maxxy = index.track_xy.max(axis=0) + 150
    road_ids = sorted(map(int, index.tree.query(box(*minxy, *maxxy))))
    # Sampling affects the background display only; candidate queries use all arcs.
    if len(road_ids) > max_roads:
        road_ids = np.asarray(road_ids)[np.linspace(0, len(road_ids)-1, max_roads, dtype=int)].tolist()
    roads = folium.FeatureGroup(name=f'Road network background ({len(road_ids)} arcs shown)', show=False)
    for arc in road_ids:
        coords = trip.nodes[trip.arcs[arc]][:, ::-1].tolist()
        folium.PolyLine(coords, color='#777777', weight=2, opacity=.5).add_to(roads)
    roads.add_to(result)
    if trip.route is not None:
        reference = folium.FeatureGroup(name='Reference route (development only)')
        for arc in trip.route:
            folium.PolyLine(trip.nodes[trip.arcs[arc]][:, ::-1].tolist(), color='green', weight=4, opacity=.7).add_to(reference)
        reference.add_to(result)
    gps = folium.FeatureGroup(name='Raw GPS observations (sampled display)')
    sample = np.unique(np.linspace(0, len(trip.track)-1, min(max_points, len(trip.track)), dtype=int))
    for i in sample:
        lon, lat, timestamp = trip.track[i]
        folium.CircleMarker([lat, lon], radius=2, color='blue', fill=True,
                            tooltip=f'Observation {i}; time {timestamp:g}s').add_to(gps)
    gps.add_to(result)
    result.fit_bounds([[float(trip.track[:, 1].min()), float(trip.track[:, 0].min())],
                       [float(trip.track[:, 1].max()), float(trip.track[:, 0].max())]])
    folium.LayerControl().add_to(result)
    result.save(str(output))
    output = Path(output)
    output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines()) + '\n')


def run_inventory(root, output_dir, *, map_trip=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root, output = Path(root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, errors, selected = [], [], None
    partitions = ['train', 'student_test'] if (root / 'train').is_dir() else [None]
    for partition in partitions:
        for directory in trip_directories(root / partition if partition else root):
            try:
                trip = load_trip(directory, include_reference=True)
                row = inspect_trip(trip)
                row['partition'] = partition or root.name
                row['bytes'] = sum(p.stat().st_size for p in directory.iterdir() if p.suffix in {'.track','.nodes','.arcs','.route'})
                rows.append(row)
                if selected is None and (map_trip is None or trip.trip_id == map_trip):
                    selected = trip
            except ValueError as exc:
                errors.append(dict(trip_id=directory.name, partition=partition, error=str(exc)))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'dataset_inventory.csv', index=False)
    (output / 'validation_errors.json').write_text(json.dumps(errors, indent=2))
    if len(frame):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for name, group in frame.groupby('partition'):
            axes[0].hist(group.observations, bins=20, alpha=.5, label=name)
            axes[1].scatter(group.observations, group.arcs, s=15, label=name)
            axes[2].scatter(group.observations, group.max_gap_s.clip(lower=1), s=15, label=name)
        axes[0].set(xlabel='GPS observations per trip', ylabel='Number of trips')
        axes[1].set(xlabel='GPS observations', ylabel='Road arcs')
        axes[2].set(xlabel='GPS observations', ylabel='Largest timestamp gap (s)', yscale='log')
        for axis in axes: axis.legend()
        fig.tight_layout();fig.savefig(output / 'dataset_summary.png', dpi=150);plt.close(fig)
    if selected is not None:
        make_map(selected, output / f'trip_{selected.trip_id}.html')
        fig, axis = plt.subplots(figsize=(9, 6))
        for sequence_idx, arc in enumerate(selected.route):
            coords = selected.nodes[selected.arcs[arc]]
            axis.plot(coords[:, 0], coords[:, 1], color='green', alpha=.7, linewidth=2,
                      label='Reference arcs' if sequence_idx == 0 else None)
        axis.scatter(selected.track[:, 0], selected.track[:, 1], s=2, color='blue', alpha=.5, label='GPS observations')
        axis.set(xlabel='Longitude (degrees)', ylabel='Latitude (degrees)', title=f'Development trip {selected.trip_id}: GPS and reference arcs')
        axis.legend();fig.tight_layout();fig.savefig(output / 'representative_trip.png', dpi=150);plt.close(fig)
    print(json.dumps(dict(valid_trips=len(rows), errors=len(errors), output=str(output))))
    if errors:
        raise ValueError(f'{len(errors)} invalid trips; see validation_errors.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--map-trip')
    args = parser.parse_args()
    run_inventory(args.input_dir, args.output_dir, map_trip=args.map_trip)


if __name__ == '__main__':
    main()
