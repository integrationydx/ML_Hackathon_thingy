"""Known-geometry and leakage regression tests; no external dataset required."""
from pathlib import Path
import numpy as np
import pytest
from src.data.loaders import load_trip, DataValidationError
from src.features.spatial import RoadIndex
from src.features.build_features import prepare_trip


def write_trip(tmp_path: Path, *, route='0\n1\n', times=(1, 2, 3)):
    trip = tmp_path / '00000001'
    trip.mkdir()
    (trip / '00000001.nodes').write_text('8 49\n8.001 49\n8.002 49\n')
    (trip / '00000001.arcs').write_text('0 1\n1 2\n1 0\n')
    (trip / '00000001.track').write_text('\n'.join(
        f'{lon} 49.00001 {t}' for lon, t in zip([8.0002, 8.0008, 8.0018], times)))
    (trip / '00000001.route').write_text(route)
    return trip


def test_loaders_preserve_row_ids_and_route_is_not_point_labels(tmp_path):
    trip = load_trip(write_trip(tmp_path), include_reference=True)
    assert trip.track.shape == (3, 3)
    assert trip.route.tolist() == [0, 1]
    assert trip.arcs.tolist() == [[0, 1], [1, 2], [1, 0]]


@pytest.mark.parametrize('times', [(1, 1, 3), (3, 2, 1)])
def test_rejects_nonincreasing_timestamps(tmp_path, times):
    with pytest.raises(DataValidationError, match='timestamp'):
        load_trip(write_trip(tmp_path, times=times))


def test_rejects_invalid_graph_reference(tmp_path):
    directory = write_trip(tmp_path)
    (directory / '00000001.arcs').write_text('0 999\n')
    with pytest.raises(DataValidationError, match='node'):
        load_trip(directory)


def test_rejects_fractional_integer_ids(tmp_path):
    directory = write_trip(tmp_path)
    (directory / '00000001.arcs').write_text('0 1.5\n')
    with pytest.raises(DataValidationError, match='integer'):
        load_trip(directory)


def test_candidate_distance_heading_and_projection(tmp_path):
    trip = load_trip(write_trip(tmp_path))
    index = RoadIndex(trip)
    choices = index.candidates(trip.track[0, :2], radius_m=10, max_candidates=8)
    assert {c['arc_id'] for c in choices} == {0, 2}
    assert 0.5 < choices[0]['distance_m'] < 2.0
    assert choices[0]['projection_ratio'] == pytest.approx(0.2, abs=0.01)
    assert index.candidates([8, 49.1], radius_m=10, max_candidates=8) == []


def test_inference_does_not_read_reference_or_change_with_labels(tmp_path):
    directory = write_trip(tmp_path)
    before = prepare_trip(load_trip(directory), radius_m=20)
    (directory / '00000001.route').write_text('not a valid route file')
    after = prepare_trip(load_trip(directory), radius_m=20)
    assert before.candidates.equals(after.candidates)
    assert before.labels is None and after.labels is None
    assert 'reference_arc_id' not in after.candidates
    assert 'target' not in after.candidates


def test_alignment_is_separate_monotone_and_gap_aware(tmp_path):
    trip = load_trip(write_trip(tmp_path, times=(1, 2, 200)), include_reference=True)
    output = prepare_trip(trip, radius_m=20, gap_s=60, include_labels=True)
    assert output.labels.reference_arc_id.tolist() == [0, 0, 1]
    assert output.observations.segment_id.tolist() == [0, 0, 1]
    assert output.candidates.groupby('obs_idx').candidate_idx.apply(list).tolist() == [[0, 1], [0, 1, 2], [0]]
    assert output.labels.candidate_miss.tolist() == [False, False, False]
    assert np.isfinite(output.candidates.select_dtypes(include='number')).all().all()


def test_candidate_misses_are_retained_as_observations(tmp_path):
    directory = write_trip(tmp_path)
    (directory / '00000001.track').write_text('8 49.1 1\n')
    result = prepare_trip(load_trip(directory, include_reference=True), radius_m=1, include_labels=True)
    assert len(result.observations) == 1
    assert result.observations.num_candidates.tolist() == [0]
    assert len(result.candidates) == 0
    assert result.labels.alignment_status.tolist() == ['unmatched']


def test_export_refuses_existing_output_and_writes_typed_empty_tables(tmp_path):
    from src.features.build_features import run_partition
    import pandas as pd
    partition = tmp_path / 'train'
    partition.mkdir()
    directory = write_trip(partition)
    (directory / '00000001.track').write_text('8 49.1 1\n')
    output = tmp_path / 'prepared'
    run_partition(partition, output, include_labels=True, radius_m=1)
    assert pd.read_parquet(output / 'candidates.parquet').empty
    assert pd.read_parquet(output / 'train_features.parquet').empty
    with pytest.raises(FileExistsError):
        run_partition(partition, output, include_labels=False)


def test_failed_export_has_no_success_manifest(tmp_path):
    from src.features.build_features import run_partition
    partition = tmp_path / 'train'
    partition.mkdir()
    directory = write_trip(partition)
    (directory / '00000001.track').write_text('8 49 1\n8 49 1\n')
    output = tmp_path / 'prepared'
    with pytest.raises(DataValidationError):
        run_partition(partition, output)
    assert not (output / 'manifest.json').exists()


@pytest.mark.parametrize('parameter', ['radius_m', 'gap_s', 'reference_radius_m'])
def test_rejects_nonfinite_preparation_limits(tmp_path, parameter):
    trip = load_trip(write_trip(tmp_path))
    with pytest.raises(ValueError):
        prepare_trip(trip, **{parameter: float('nan')})
