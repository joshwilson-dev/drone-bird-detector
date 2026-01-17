# tiling.py

import math
from typing import List, Tuple, Dict
from PIL import Image
import torchvision.transforms.functional as F
import torch

def compute_grid(
    image_width: int,
    image_height: int,
    tile_width: int,
    tile_height: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    """
    Compute top-left (row, col) coordinates for overlapping tiles.
    """
    overlap_w = min(overlap, tile_width - 1)
    overlap_h = min(overlap, tile_height - 1)

    step_x = tile_width - overlap_w
    step_y = tile_height - overlap_h

    xs = list(range(0, max(image_width - tile_width, 0) + 1, step_x))
    ys = list(range(0, max(image_height - tile_height, 0) + 1, step_y))

    # Ensure last tile reaches image boundary
    if xs[-1] + tile_width < image_width:
        xs.append(image_width - tile_width)
    if ys[-1] + tile_height < image_height:
        ys.append(image_height - tile_height)

    return [(y, x) for y in ys for x in xs]


def tile_image(
    image: Image.Image,
    tile_width: int,
    tile_height: int,
    overlap: int,
    to_tensor: bool = True,
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Split an image into overlapping tiles.

    Returns
    -------
    tiles : list of tensors or PIL Images
    metadata : list of dicts containing tile position info
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_w, img_h = image.size
    grid = compute_grid(
        image_width=img_w,
        image_height=img_h,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
    )

    tiles = []
    metadata = []

    for idx, (top, left) in enumerate(grid):
        tile = F.crop(image, top, left, tile_height, tile_width)

        if to_tensor:
            tile = F.to_tensor(tile)

        tiles.append(tile)
        metadata.append(
            {
                "tile_id": idx,
                "top": top,
                "left": left,
                "height": img_h,
                "width": img_w,
            }
        )

    return tiles, metadata