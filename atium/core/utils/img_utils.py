
import cv2
import numpy as np

from atium.core.utils.custom_types import BGRColor, CoordinateXY, ImageArray3D, Index2D


def paint_img_inplace(
    img: ImageArray3D,
    color: BGRColor,
) -> None:
    img[:, :, 0] = color[0]
    img[:, :, 1] = color[1]
    img[:, :, 2] = color[2]


def create_canvas(
    img_width: int,
    img_height: int,
    color: BGRColor | None = None,
) -> ImageArray3D:
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    if color is not None:
        paint_img_inplace(
            img=img,
            color=color,
        )
    return img


def world_coordinate_to_px_coordinate(
    canvas: ImageArray3D,
    world_xy: CoordinateXY,
    resolution: float,
) -> Index2D:
    return (
        int(world_xy[0] // resolution),
        canvas.shape[0] - int(world_xy[1] // resolution),
    )


def draw_line_on_canvas(
    canvas: ImageArray3D,
    start_xy: CoordinateXY,
    end_xy: CoordinateXY,
    color: BGRColor,
    thickness_px: int,
    resolution: float,
) -> None:
    """
    Draws a line on the canvas.
    Start and end points are in meters assuming an x-y coordinate frame origin
    on the bottom left corner of the image.
    """
    start_px = world_coordinate_to_px_coordinate(
        canvas=canvas,
        world_xy=start_xy,
        resolution=resolution,
    )
    end_px = world_coordinate_to_px_coordinate(
        canvas=canvas,
        world_xy=end_xy,
        resolution=resolution,
    )

    cv2.line(
        canvas,
        start_px,
        end_px,
        color,
        thickness_px,
    )


def draw_rectangle_on_canvas(
    canvas: ImageArray3D,
    center_xy: CoordinateXY,
    length: float,
    width: float,
    color: BGRColor,
    thickness_px: int,
    resolution: float,
) -> None:
    """
    Draws a rectangle on the canvas.
    Center and dimensions are in world coordinates.
    """
    top_left_xy = (
        center_xy[0] - 0.5 * length,
        center_xy[1] + 0.5 * width,
    )
    bottom_right_xy = (
        center_xy[0] + 0.5 * length,
        center_xy[1] - 0.5 * width,
    )
    top_left_px = world_coordinate_to_px_coordinate(
        canvas=canvas,
        world_xy=top_left_xy,
        resolution=resolution,
    )
    bottom_right_px = world_coordinate_to_px_coordinate(
        canvas=canvas,
        world_xy=bottom_right_xy,
        resolution=resolution,
    )
    cv2.rectangle(
        canvas,
        top_left_px,
        bottom_right_px,
        color,
        thickness_px,
    )


def draw_circle_on_canvas(
    canvas: ImageArray3D,
    center_xy: CoordinateXY,
    radius: float,
    color: BGRColor,
    thickness_px: int,
    resolution: float,
) -> None:
    """
    Draws a circle on the canvas.
    Center and dimensions are in world coordinates.
    """
    center_px = world_coordinate_to_px_coordinate(
        canvas=canvas,
        world_xy=center_xy,
        resolution=resolution,
    )
    radius_px = int(radius // resolution)
    cv2.circle(
        canvas,
        center_px,
        radius_px,
        color,
        thickness_px,
    )
