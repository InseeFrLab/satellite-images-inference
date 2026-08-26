import albumentations as A
import cv2
import numpy as np
import torch
from astrovision.data import SatelliteImage
from astrovision.data.utils import get_bounds_for_tile, get_transform_for_tile
from monai.inferers import SlidingWindowSplitter

from app.logger_config import configure_logger
from app.utils.data import get_satellite_image

logger = configure_logger()

# ---- Master function ---- #


def get_normalized_sis(
    image: str,
    n_bands: int,
    tiles_size: int,
    normalization_mean: list[float],
    transform: A.Compose,
    sliding_window_split: bool,
    **kwargs,
) -> tuple[torch.Tensor, list[SatelliteImage]]:
    """
    Retrieves and normalizes satellite images based on the specified parameters.

    Args:
        image (str): The input image file to be processed.
        n_bands (int): Number of bands in the satellite image.
        tiles_size (int): Size of the tiles for splitting the image.
        normalization_mean (List[float]): Mean values for normalization.
        sliding_window (bool): If True, uses sliding window approach; otherwise, uses classic approach.
        transform (albumentations.Compose): Transformation to apply to the image.
        kwargs: Additional keyword arguments for the sliding window splitter.

    Returns:
        torch.Tensor: Normalized satellite images as a tensor.
        list[SatelliteImage]: List of SatelliteImage objects representing the split tiles.
    """

    si = get_satellite_image(image, n_bands)  # get SatelliteImage object from the image path

    # Deal when images to pred have more channels than images used during training
    if len(normalization_mean) != si.array.shape[0]:
        si.array = si.array[: len(normalization_mean)]

    if si.array.shape[1] % tiles_size != 0:
        raise ValueError("The dimension of the image must be divisible by the tiles size used during training.")
    if si.array.shape[1] <= tiles_size:
        raise ValueError("The dimension of the image should be equal to or greater than the tile size used during training.")

    # Normalize image if it is not in uint8
    if si.array.dtype is not np.dtype("uint8"):
        si.array = cv2.normalize(si.array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if sliding_window_split:
        if kwargs.get("overlap") is None:
            raise ValueError("If sliding_window_split is set to True, overlap must be specified.")
        return _get_normalized_sis_sliding_window(si, tiles_size, transform, **kwargs)
    else:
        if kwargs:
            logger.info(
                f"sliding_window_split is set to False, using classic split method; and ignoring the following parameters: {kwargs}"
            )
        return _get_normalized_sis_classic(si, tiles_size, transform)


# ---- Helper private functions ---- #
# Both of them return:
# - normalized_sis_tensor: tensor shape (num_tiles, n_bands, augment_size, augment_size)
# - si_splitted: list of SatelliteImage objects representing the split tiles (length num_tiles)


def _get_normalized_sis_classic(
    si: SatelliteImage, tiles_size: int, transform: A.Compose
) -> tuple[torch.Tensor, list[SatelliteImage]]:
    """
    Splits the SatelliteImage object into smaller tiles and normalizes each tile.
    Using "classic" split from astrovision.

    Args:
        si (SatelliteImage): The SatelliteImage object to be split and normalized. Output of get_satellite_image.
        tiles_size (int): The size of the tiles to split the image into.
        transform (albumentations.Compose): The transformation to apply to each tile. Output of get_transform.
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: A tensor containing the normalized tiles.
            - list[SatelliteImage]: A list of SatelliteImage objects representing the split tiles.
    """

    si_splitted = si.split(tiles_size)  # split in a list of smaller SatelliteImage objects (tiles)

    # Each tile is normalized + converted to tensor
    normalized_sis = [
        transform(image=np.transpose(s_si.array, [1, 2, 0]))["image"].unsqueeze(dim=0) for s_si in si_splitted
    ]  # list of tensors of size (n_bands, augment_size, augment_size) ; length is number of tiles
    normalized_sis_tensor = torch.vstack(normalized_sis)  # tensor shape (num_tiles, n_bands, augment_size, augment_size)

    return normalized_sis_tensor, si_splitted


def _get_normalized_sis_sliding_window(
    si: SatelliteImage, tiles_size: int, transform: A.Compose, overlap: int, pad_mode: str = "constant", pad_value: int = 0
):
    """
    Splits the SatelliteImage object into smaller tiles using a sliding window approach and normalizes each tile.
    Uses SlidingWindowSplitter from MONAI.

    Args:
        si (SatelliteImage): The SatelliteImage object to be split and normalized. Output of get_satellite_image.
        tiles_size (int): The size of the tiles to split the image into.
        transform (albumentations.Compose): The transformation to apply to each tile. Output of get_transform.

        SlidingWindowSplitter params:
            overlap (int): The overlap size between the tiles.
            pad_mode (str): The padding mode to use for the sliding window. Defaults to "constant".
            pad_value (int): The value to use for padding. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: A tensor containing the normalized tiles.
            - list[SatelliteImage]: A list of SatelliteImage objects representing the split tiles.

    """
    splitter = SlidingWindowSplitter(patch_size=(tiles_size, tiles_size), overlap=overlap, pad_mode=pad_mode, pad_value=pad_value)
    split_input = splitter(torch.from_numpy(si.array).unsqueeze(0))

    normalized_sis = []
    si_splitted = []
    for i, (patch, loc) in enumerate(split_input):
        rows = (loc[0], loc[0] + tiles_size)
        cols = (loc[1], loc[1] + tiles_size)
        array = patch.squeeze(0).numpy()

        s_si = SatelliteImage(
            array=array,
            crs=si.crs,
            bounds=get_bounds_for_tile(si.transform, rows, cols),
            transform=get_transform_for_tile(si.transform, rows[0], cols[0]),
            dep=si.dep,
            date=si.date,
        )
        si_splitted.append(s_si)
        normalized_si = transform(image=array.transpose(1, 2, 0))["image"].unsqueeze(dim=0)
        normalized_sis.append(normalized_si)

    normalized_sis_tensor = torch.vstack(normalized_sis)

    return normalized_sis_tensor, si_splitted
