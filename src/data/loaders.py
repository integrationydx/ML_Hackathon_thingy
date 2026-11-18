"""Strict raw parsing. Reference routes are opt-in and never needed for inference."""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


class DataValidationError(ValueError):
    """A raw trip violates the declared input contract."""


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None = None


def _read(path: Path, columns: int, *, integer: bool = False) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise DataValidationError(f'{path}: cannot parse {columns} numeric columns') from exc
    if values.size == 0 or values.shape[1] != columns:
        raise DataValidationError(f'{path}: expected nonempty {columns}-column data')
    if not np.isfinite(values).all():
        raise DataValidationError(f'{path}: nonfinite values')
    if integer:
        if (values < 0).any() or (values >= 2**63).any() or not np.equal(values, np.floor(values)).all():
            raise DataValidationError(f'{path}: expected nonnegative integer IDs')
        return values.astype(np.int64)
    return values


def load_trip(directory: str | Path, *, include_reference: bool = False) -> Trip:
    directory = Path(directory)
    trip_id = directory.name
    if not re.fullmatch(r'[0-9]{8}', trip_id):
        raise DataValidationError(f'{directory}: expected eight-digit trip ID')
    track = _read(directory / f'{trip_id}.track', 3)
    nodes = _read(directory / f'{trip_id}.nodes', 2)
    arcs = _read(directory / f'{trip_id}.arcs', 2, integer=True)
    for name, coords in [('track', track[:, :2]), ('nodes', nodes)]:
        if (np.abs(coords[:, 0]) > 180).any() or (np.abs(coords[:, 1]) > 90).any():
            raise DataValidationError(f'{trip_id}: {name} coordinates outside lon/lat bounds')
    if (np.diff(track[:, 2]) <= 0).any():
        raise DataValidationError(f'{trip_id}: timestamps must increase strictly')
    if (arcs >= len(nodes)).any():
        raise DataValidationError(f'{trip_id}: arc references nonexistent node')
    route = None
    if include_reference:
        route = _read(directory / f'{trip_id}.route', 1, integer=True).ravel()
        if (route >= len(arcs)).any():
            raise DataValidationError(f'{trip_id}: route references nonexistent arc')
    return Trip(trip_id, track, nodes, arcs, route)


def trip_directories(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise DataValidationError(f'Partition directory does not exist: {root}')
    directories = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'[0-9]{8}', p.name))
    if not directories:
        raise DataValidationError(f'No eight-digit trip directories in {root}')
    return directories
