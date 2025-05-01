from enum import IntEnum
from functools import cached_property
from typing import Self

import attr
import cv2
import numpy as np

from atium.core.utils.attrs_utils import AttrsValidators
from atium.core.utils.color_utils import AtiumColorsBGR, ColorType
from atium.core.utils.custom_types import (
    BGRColor,
    CoordinateXY,
    DistanceMap2D,
    EnvironmentArray2D,
    ImageArray3D,
    Index2D,
    Shape2D,
    SizeXY,
)


class EnvironmentLabels(IntEnum):
    FREE = 0
    STATIC_OBSTACLE = 1

    def get_viz_color(self, color_type: ColorType = ColorType.BGR) -> BGRColor:
        bgr_color = {
            EnvironmentLabels.FREE: AtiumColorsBGR.WHITE,
            EnvironmentLabels.STATIC_OBSTACLE: AtiumColorsBGR.PUMPKIN,
        }[self]
        if color_type == ColorType.RGB:
            return bgr_color[::-1]
        return bgr_color


@attr.define
class EnvironmentMap2D:
    """
    A map of obstacles in the environment represented as a numpy array.
    """

    array: EnvironmentArray2D = attr.ib(
        validator=AttrsValidators.array_2d_validator(
            desired_dtype=np.uint8,
        )
    )
    resolution: float = attr.ib(validator=AttrsValidators.scalar_bounding_box_validator(min_value=0.0, inclusive=False))

    @classmethod
    def from_empty(cls, size_xy: SizeXY, resolution: float) -> Self:
        size_px = (int(size_xy[1] / resolution), int(size_xy[0] / resolution))
        return cls(
            array=np.zeros(size_px, dtype=np.uint8),
            resolution=resolution,
        )

    # Properties
    # --------------------------------------------------

    @cached_property
    def size_px(self) -> Shape2D:
        return self.array.shape

    @cached_property
    def size_xy(self) -> SizeXY:
        return (self.size_px[1] * self.resolution, self.size_px[0] * self.resolution)

    # Coordinates/indices
    # --------------------------------------------------

    def xy_to_px(self, xy: tuple[float, float], output_as_float: bool = False) -> tuple[int, int] | tuple[float, float]:
        """
        Convert a tuple (can be size, coordinates, etc) in meters to a tuple in pixels.
        """
        if output_as_float:
            return (
                xy[0] / self.resolution,
                xy[1] / self.resolution,
            )
        return (
            int(xy[0] / self.resolution),
            int(xy[1] / self.resolution),
        )

    def px_to_xy(self, px: tuple[int, int]) -> tuple[float, float]:
        """
        Convert a tuple (can be size, coordinates, etc) in pixels to a tuple in meters.
        """
        return (
            px[0] * self.resolution,
            px[1] * self.resolution,
        )

    def get_cv2_coordinates(self, position_px: Index2D) -> Index2D:
        """
        Convert a pixel coordinate to a cv2 coordinate.
        The cv2 coordinate system has the origin at the top-left corner of the image.
        """
        return (position_px[0], self.size_px[0] - position_px[1])

    # Obstacles
    # --------------------------------------------------

    def add_rectangular_obstacle(
        self,
        center_xy: CoordinateXY,
        size_xy: SizeXY,
        label: EnvironmentLabels,
        thickness: int = cv2.FILLED,
    ) -> None:
        """
        Draw a rectangle on the map.
        The rectangle is defined by its center and size.
        """
        # Convert position to pixel coordinates
        center_px = self.xy_to_px(xy=center_xy)
        size_px = self.xy_to_px(xy=size_xy)

        # Calculate the top-left and bottom-right corners of the rectangle
        top_left = (center_px[0] - size_px[0] // 2, center_px[1] - size_px[1] // 2)
        bottom_right = (center_px[0] + size_px[0] // 2, center_px[1] + size_px[1] // 2)

        cv2.rectangle(
            self.array,
            self.get_cv2_coordinates(top_left),
            self.get_cv2_coordinates(bottom_right),
            color=label,
            thickness=thickness,
        )

    # Computation
    # --------------------------------------------------

    def compute_signed_distance_transform(self) -> DistanceMap2D:
        """
        Compute the signed distance transform of the map.
        The signed distance transform is a map where each pixel value is the distance to the nearest obstacle.
        The distance is positive if the pixel is inside an obstacle and negative if it is outside.
        To compute this, we subtract two distance transforms:
        1. The distance transform of the free space to obstacles.
        2. The distance transform of the obstacles to free space.
        """
        dtf_input = np.zeros_like(self.array, dtype=np.uint8)
        dtf_input[self.array == EnvironmentLabels.FREE] = 1
        dtf_free_map = cv2.distanceTransform(
            src=dtf_input,
            distanceType=cv2.DIST_L2,
            maskSize=cv2.DIST_MASK_PRECISE,
            dstType=cv2.CV_32F,
        )
        dtf_obstacle_map = cv2.distanceTransform(
            src=1 - dtf_input,
            distanceType=cv2.DIST_L2,
            maskSize=cv2.DIST_MASK_PRECISE,
            dstType=cv2.CV_32F,
        )
        return self.resolution * (dtf_free_map - dtf_obstacle_map)

    # Visualization
    # --------------------------------------------------

    def create_rgb_viz(
        self,
        color_type: ColorType = ColorType.BGR,
    ) -> ImageArray3D:
        """
        Returns an RGB image represenatiion of the environment map.
        """
        rgb_img = np.zeros(
            shape=(
                self.size_px[0],
                self.size_px[1],
                3,
            ),
            dtype=np.uint8,
        )
        for label in EnvironmentLabels:
            rgb_img[self.array == label] = label.get_viz_color(
                color_type=color_type,
            )

        return rgb_img
