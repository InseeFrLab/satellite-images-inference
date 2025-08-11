import cv2
import numpy as np
import torch
from monai.inferers import SlidingWindowSplitter

from app.utils.data import get_satellite_image


def get_normalized_sis(
    image,
    n_bands,
    tiles_size,
    normalization_mean,
    transform,
    sliding_window_split: bool,
    **kwargs,
):
    """
    Retrieves and normalizes satellite images based on the specified parameters.

    Args:
        image: The input image to be processed.
        n_bands (int): Number of bands in the satellite image.
        tiles_size (int): Size of the tiles for splitting the image.
        normalization_mean (List[float]): Mean values for normalization.
        sliding_window (bool): If True, uses sliding window approach; otherwise, uses classic approach.
        transform (albumentations.Compose): Transformation to apply to the image.
        kwargs: Additional keyword arguments for the sliding window splitter.

    Returns:
        torch.Tensor: Normalized satellite images as a tensor.
    """

    si = get_satellite_image(image, n_bands)

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
        return _get_normalized_sis_sliding_window(si, tiles_size, transform, return_locs_and_patches=False, **kwargs)
    else:
        return _get_normalized_sis_classic(si, tiles_size, transform)


def _get_normalized_sis_classic(si, tiles_size, transform):
    si_splitted = si.split(tiles_size)  # split in a list of smaller SatelliteImage objects (tiles)

    # Each tile is normalized + converted to tensor
    normalized_sis = [
        transform(image=np.transpose(s_si.array, [1, 2, 0]))["image"].unsqueeze(dim=0) for s_si in si_splitted
    ]  # list of tensors of size (n_bands, augment_size, augment_size) ; length is number of tiles
    normalized_sis_tensor = torch.vstack(normalized_sis)  # tensor shape (num_tiles, n_bands, augment_size, augment_size)

    return normalized_sis_tensor, si_splitted, None


def _get_normalized_sis_sliding_window(si, tiles_size, transform, overlap, return_locs_and_patches=False, **kwargs):
    pad_mode = kwargs.get("pad_mode", "constant")  # Default to 'constant' if not provided
    pad_value = kwargs.get("pad_value", 0)  # Default to 0 if not provided

    splitter = SlidingWindowSplitter(patch_size=(tiles_size, tiles_size), overlap=overlap, pad_mode=pad_mode, pad_value=pad_value)
    split_input = splitter(torch.from_numpy(si.array).unsqueeze(0))

    normalized_sis = []
    patches = []
    locs = []
    for i, (patch, loc) in enumerate(split_input):
        if return_locs_and_patches:
            patches.append(patch)
            locs.append(loc)
        normalized_si = transform(image=patch.squeeze(0).numpy().transpose(1, 2, 0))["image"].unsqueeze(dim=0)
        normalized_sis.append(normalized_si)

    normalized_sis_tensor = torch.vstack(normalized_sis)

    if return_locs_and_patches:
        return normalized_sis_tensor, patches, locs
