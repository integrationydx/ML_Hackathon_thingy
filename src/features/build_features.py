"""Prepare GPS observations and road candidates for PathSnap inference.

Reference-route labels are added only when the development-only option is enabled.
"""
import argparse
from dataclasses import dataclass
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import Trip, load_trip, trip_directories
from src.features.spatial import RoadIndex, angle_difference

FEATURE_COLUMNS = ['distance_m', 'heading_delta_deg', 'projection_ratio',
                   'distance_rank', 'segment_length_m', 'speed_mps', 'dt_s',
                   'heading_valid', 'gap_before', 'zero_length_arc']
CANDIDATE_COLUMNS = ['trip_id', 'obs_idx', 'timestamp_s', 'segment_id', 'candidate_idx',
                     'arc_id', 'from_node_id', 'to_node_id', 'projected_x_m', 'projected_y_m'] + FEATURE_COLUMNS


@dataclass
class PreparedTrip:
    candidates: pd.DataFrame
    observations: pd.DataFrame
    labels: pd.DataFrame | None
    road_segments: pd.DataFrame
    projection: str


def prepare_trip(trip: Trip, *, radius_m=50.0, max_candidates=8, gap_s=60.0,
                 include_labels=False, reference_radius_m=100.0) -> PreparedTrip:
    """Build observation, candidate, road-segment, and optional label tables."""
    if (not np.isfinite([radius_m, gap_s, reference_radius_m]).all()
            or min(radius_m, gap_s, reference_radius_m) <= 0
            or isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
            or max_candidates < 1):
        raise ValueError('Candidate limits, radii and gap_s must be positive')
    index = RoadIndex(trip)
    dt = np.r_[0.0, np.diff(trip.track[:, 2])]
    displacement = np.vstack([np.zeros(2), np.diff(index.track_xy, axis=0)])
    distance = np.linalg.norm(displacement, axis=1)
    gaps = dt > gap_s
    headings = (np.degrees(np.arctan2(displacement[:, 0], displacement[:, 1])) + 360) % 360
    heading_valid = (distance >= 3.0) & (dt > 0) & ~gaps
    # A long GPS gap makes the displacement unsuitable for estimating speed.
    speed = np.divide(distance, dt, out=np.zeros_like(dt), where=(dt > 0) & ~gaps)
    observations = pd.DataFrame(dict(trip_id=trip.trip_id, obs_idx=np.arange(len(dt)),
        longitude=trip.track[:, 0], latitude=trip.track[:, 1], timestamp_s=trip.track[:, 2],
        segment_id=np.cumsum(gaps), dt_s=dt, gap_before=gaps,
        heading_deg=headings, heading_valid=heading_valid, speed_mps=speed,
        speed_valid=(dt > 0) & ~gaps, speed_outlier=speed > 70))
    rows = []
    counts = []
    for obs, xy in zip(observations.itertuples(), index.track_xy):
        choices = index.query_xy(xy, radius_m=radius_m, max_candidates=max_candidates)
        counts.append(len(choices))
        for rank, choice in enumerate(choices):
            arc = choice['arc_id']
            rows.append(dict(choice, trip_id=trip.trip_id, obs_idx=obs.obs_idx,
                timestamp_s=obs.timestamp_s, segment_id=obs.segment_id,
                candidate_idx=rank, from_node_id=int(trip.arcs[arc, 0]),
                to_node_id=int(trip.arcs[arc, 1]),
                heading_delta_deg=float(angle_difference(obs.heading_deg, index.headings[arc])) if obs.heading_valid else 0.0,
                distance_rank=rank + 1, segment_length_m=float(index.lengths[arc]),
                speed_mps=obs.speed_mps, dt_s=obs.dt_s, heading_valid=bool(obs.heading_valid),
                gap_before=bool(obs.gap_before), zero_length_arc=bool(index.lengths[arc] == 0)))
    observations['num_candidates'] = counts
    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    # Keep the original geographic endpoints so the road graph can be rebuilt later.
    roads = pd.DataFrame(dict(trip_id=trip.trip_id, arc_id=np.arange(len(trip.arcs)),
        from_node_id=trip.arcs[:, 0], to_node_id=trip.arcs[:, 1],
        from_lon=trip.nodes[trip.arcs[:, 0], 0], from_lat=trip.nodes[trip.arcs[:, 0], 1],
        to_lon=trip.nodes[trip.arcs[:, 1], 0], to_lat=trip.nodes[trip.arcs[:, 1], 1],
        length_m=index.lengths, heading_deg=index.headings))
    labels = None
    if include_labels:
        from src.data.alignment import align_reference
        labels = align_reference(index, observations, radius_m=reference_radius_m)
        candidate_arcs = candidates.groupby('obs_idx').arc_id.agg(set).to_dict() if len(candidates) else {}
        # If the reference is unknown, leave candidate_miss empty rather than
        # treating it as a confirmed miss.
        labels['candidate_miss'] = pd.array([
            None if pd.isna(r.reference_arc_id) else int(r.reference_arc_id) not in candidate_arcs.get(r.obs_idx, set())
            for r in labels.itertuples()], dtype='boolean')
        labels['training_eligible'] = (labels.alignment_status == 'aligned') & ~labels.candidate_miss.fillna(True)
    return PreparedTrip(candidates, observations, labels, roads, index.crs.to_string())


def training_rows(prepared):
    """Add reference labels for development without changing inference output."""
    if prepared.labels is None:
        raise ValueError('Training rows require development labels')
    merged = prepared.candidates.merge(prepared.labels, on=['trip_id', 'obs_idx'], validate='many_to_one')
    merged['target'] = (merged.arc_id == merged.reference_arc_id).fillna(False).astype('int8')
    return merged[merged.training_eligible].copy()


def run_partition(data_dir, output, *, radius_m=50, max_candidates=8, gap_s=60,
                  include_labels=False, reference_radius_m=100, limit=None):
    """Process a dataset partition and save its tables and manifest."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f'Use a new output directory; refusing stale data in {destination}')
    trips = trip_directories(data_dir)
    if limit is not None:
        if limit <= 0:
            raise ValueError('limit must be positive')
        trips = trips[:limit]
    # Write each trip as it is processed instead of keeping the entire dataset
    # and all candidate rows in memory.
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    writers = {}
    stats = []
    projection_records = {}
    try:
        for trip_dir in trips:
            trip = load_trip(trip_dir, include_reference=include_labels)
            prepared = prepare_trip(trip, radius_m=radius_m, max_candidates=max_candidates,
                gap_s=gap_s, include_labels=include_labels, reference_radius_m=reference_radius_m)
            tables = dict(candidates=prepared.candidates, observations=prepared.observations,
                          road_segments=prepared.road_segments)
            if include_labels:
                tables['reference_labels'] = prepared.labels
                tables['train_features'] = training_rows(prepared)
            for name, frame in tables.items():
                # Set the column types explicitly, including when a table is empty.
                integer_columns = {'obs_idx', 'segment_id', 'candidate_idx', 'arc_id',
                    'from_node_id', 'to_node_id', 'distance_rank', 'num_candidates',
                    'reference_arc_id', 'route_position', 'target'}
                boolean_columns = {'heading_valid', 'gap_before', 'zero_length_arc',
                    'speed_valid', 'speed_outlier', 'alignment_restart', 'candidate_miss',
                    'training_eligible'}
                string_columns = {'trip_id', 'alignment_status', 'label_method'}
                schema = pa.schema([(col, pa.int64() if col in integer_columns else
                    pa.bool_() if col in boolean_columns else pa.string() if col in string_columns
                    else pa.float64()) for col in frame.columns])
                table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
                if name not in writers:
                    writers[name] = pq.ParquetWriter(directory / f'{name}.parquet', schema, compression='zstd')
                writers[name].write_table(table)
            projection_records[trip.trip_id] = prepared.projection
            stat = dict(trip_id=trip.trip_id, observations=len(trip.track), candidates=len(prepared.candidates),
                        no_candidate_observations=int((prepared.observations.num_candidates == 0).sum()))
            if include_labels:
                labels = prepared.labels
                stat.update(eligible_observations=int(labels.training_eligible.sum()),
                    known_reference_observations=int(labels.reference_arc_id.notna().sum()),
                    candidate_misses=int(labels.candidate_miss.sum()),
                    alignment_status=labels.alignment_status.value_counts().to_dict())
            stats.append(stat)
            print(json.dumps(stat), flush=True)
    except BaseException:
        for writer in writers.values():
            writer.close()
        shutil.rmtree(directory)
        raise
    else:
        for writer in writers.values():
            writer.close()
    manifest = dict(schema_version=1, data_partition=Path(data_dir).name,
        label_warning='Approximate reference alignment; not independently verified point ground truth.',
        feature_columns=FEATURE_COLUMNS, radius_m=radius_m, max_candidates=max_candidates,
        gap_s=gap_s, reference_radius_m=reference_radius_m, include_labels=include_labels,
        alignment_config=dict(method='monotone_reference_v1', max_candidates=8,
            max_label_distance_m=50.0, ambiguity_margin_m=3.0,
            heading_cost_m_per_degree=0.05, route_advance_cost_m=0.01),
        projections=projection_records, trips=stats)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (directory / 'feature_columns.json').write_text(json.dumps(FEATURE_COLUMNS, indent=2))
    if destination.exists():
        shutil.rmtree(directory)
        raise FileExistsError(f'Output appeared during processing: {destination}')
    directory.rename(destination)
    return manifest


def main():
    """Read command-line options and start partition processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--radius-m', type=float, default=50)
    parser.add_argument('--max-candidates', type=int, default=8)
    parser.add_argument('--gap-s', type=float, default=60)
    parser.add_argument('--reference-radius-m', type=float, default=100)
    parser.add_argument('--include-labels', action='store_true', help='Development only: explicitly read .route')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run_partition(args.data_dir, args.output_dir, radius_m=args.radius_m,
        max_candidates=args.max_candidates, gap_s=args.gap_s, include_labels=args.include_labels,
        reference_radius_m=args.reference_radius_m, limit=args.limit)


if __name__ == '__main__':
    main()
