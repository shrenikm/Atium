from typing import Optional
import numpy as np

from common.colors import AtiumColorsBGR
from common.custom_types import BGRColor, Img3Channel


def paint_img_inplace(
    img: Img3Channel,
    color: BGRColor,
) -> None:
    img[:, :, 0] = color[0]
    img[:, :, 1] = color[1]
    img[:, :, 2] = color[2]


def create_canvas(
    img_width: int,
    img_height: int,
    color: Optional[BGRColor] = None,
) -> Img3Channel:
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    if color is not None:
        paint_img_inplace(
            img=img,
            color=color,
        )
    return img
