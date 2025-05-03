import numpy as np

from atium.core.utils.custom_types import AngleOrAnglesRad, CoordinateXY, PolygonXYArray, SizeXY


def normalize_angle(angles: AngleOrAnglesRad) -> AngleOrAnglesRad:
    # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    return (np.array(angles) + np.pi) % (2 * np.pi) - np.pi


def construct_rectangle_polygon(
    center_xy: CoordinateXY,
    size_xy: SizeXY,
) -> PolygonXYArray:
    """
    Constructs a rectangle polygon with the given size.
    The rectangle is centered at the origin and aligned with the x-y axes.
    """
    width, height = size_xy
    return (
        np.array(
            [
                [-width / 2, -height / 2],
                [width / 2, -height / 2],
                [width / 2, height / 2],
                [-width / 2, height / 2],
            ],
            dtype=np.float64,
        )
        + np.array(center_xy, dtype=np.float64)
    )


def densify_polygon(polygon: PolygonXYArray, spacing: float) -> PolygonXYArray:
    """
    Densify a polygon represented by Nx2 corner points by adding points along
    each edge at roughly `distance_between_points` intervals.
    """
    segments_start = polygon
    segments_end = np.roll(polygon, -1, axis=0)  # shift to get the next point, wrapping around

    edge_vectors = segments_end - segments_start
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    all_points = []

    for start, vec, length in zip(segments_start, edge_vectors, edge_lengths, strict=True):
        num_segments = max(int(np.floor(length / spacing)), 1)
        t_values = np.linspace(0, 1, num_segments, endpoint=False)
        points = start + np.outer(t_values, vec)
        all_points.append(points)

    return np.vstack(all_points)
