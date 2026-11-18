"""Spatial helpers for finding nearby road segments for a GPS point."""

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from src.data.loaders import Trip


class RoadIndex:
    def __init__(self, trip: Trip):
        """Build a spatial index for the roads and GPS track in one trip."""

        # Use a circular mean for longitude so trips near the antimeridian
        # are handled correctly.
        longitude = np.deg2rad(trip.track[:, 0])
        lon0 = np.rad2deg(
            np.arctan2(np.sin(longitude).mean(), np.cos(longitude).mean())
        )
        lat0 = float(np.median(trip.track[:, 1]))

        self.crs = CRS.from_proj4(
            f'+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m'
        )
        self.transformer = Transformer.from_crs(
            'EPSG:4326', self.crs, always_xy=True
        )

        self.trip = trip
        self.nodes_xy = self.project(trip.nodes)
        self.track_xy = self.project(trip.track[:, :2])

        # Get the projected start and end points for every directed road arc.
        self.starts = self.nodes_xy[trip.arcs[:, 0]]
        self.ends = self.nodes_xy[trip.arcs[:, 1]]

        vector = self.ends - self.starts
        self.lengths = np.linalg.norm(vector, axis=1)
        self.headings = (
            np.degrees(np.arctan2(vector[:, 0], vector[:, 1])) + 360
        ) % 360

        # The tree makes it faster to find road segments near a GPS point.
        self.geometries = [
            LineString([start, end])
            for start, end in zip(self.starts, self.ends)
        ]
        self.tree = STRtree(self.geometries)

    def project(self, coordinates):
        """Convert longitude-latitude coordinates into the trip's local CRS."""

        values = np.asarray(coordinates)
        x, y = self.transformer.transform(values[..., 0], values[..., 1])
        result = np.stack([x, y], axis=-1)

        if not np.isfinite(result).all():
            raise ValueError('Projection produced nonfinite coordinates')

        return result

    def candidates(self, lonlat, *, radius_m=50.0, max_candidates=8):
        """Find nearby road candidates for a longitude-latitude point."""

        return self.query_xy(
            self.project(lonlat),
            radius_m=radius_m,
            max_candidates=max_candidates,
        )

    def query_xy(self, xy, *, radius_m, max_candidates):
        """Return the closest road segments to a point in projected coordinates."""

        if (not np.isfinite(radius_m) or radius_m <= 0
                or isinstance(max_candidates, bool) or not isinstance(max_candidates, int)
                or max_candidates < 1):
            raise ValueError('radius_m and max_candidates must be positive')

        point = Point(xy)

        # Check the distance to the full road segment, not just its midpoint.
        ids = self.tree.query(
            point,
            predicate='dwithin',
            distance=radius_m,
        )
        ordered = sorted(
            (float(point.distance(self.geometries[i])), int(i))
            for i in ids
        )

        result = []
        for distance, arc_id in ordered[:max_candidates]:
            line = self.geometries[arc_id]
            length = float(self.lengths[arc_id])

            # Find where the GPS point falls along the selected road segment.
            along = float(line.project(point)) if length > 0 else 0.0
            projected = line.interpolate(along)

            result.append(
                dict(
                    arc_id=arc_id,
                    distance_m=distance,
                    projection_ratio=along / length if length > 0 else 0.0,
                    projected_x_m=float(projected.x),
                    projected_y_m=float(projected.y),
                )
            )

        return result


def angle_difference(a, b):
    """Return the smallest absolute angular difference, in degrees."""

    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)
