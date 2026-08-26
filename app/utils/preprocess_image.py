from typing import List

import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from astrovision.data import SatelliteImage


def preprocess_image(
    image: SatelliteImage,
    normalization_mean: List[float],
    transform: A.Compose,
):
    """
    Preprocesses a satellite image using the specified model.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        image (SatelliteImage): SatelliteImage object representing the input image.
        tiles_size (int): Size of the satellite image.
        augment_size (int): .
    Returns:
        torch.Tensor: Normalized and preprocessed image tensor.
    """

    # Deal when images to pred have more channels than images used during training
    if len(normalization_mean) != image.array.shape[0]:
        image.array = image.array[: len(normalization_mean)]

    # Apply transformation to image
    normalized_si = transform(image=np.transpose(image.array, [1, 2, 0]))["image"].unsqueeze(dim=0)

    return normalized_si


def get_transform(
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
):
    """
    Retrieves the transformation pipeline for image preprocessing.

    Args:
        tiles_size (int): Size of the satellite image.
        augment_size (int): .
        n_bands: (int): .
        normalization_mean List[float]: .
        normalization_std List[float]: .

    Returns:
        albumentations.Compose: A composition of image transformations.
    """
    # Define the transformation pipeline
    transform_list = [
        A.Normalize(
            max_pixel_value=1.0,
            mean=normalization_mean,
            std=normalization_std,
        ),
        ToTensorV2(),
    ]

    # Add resizing transformation if augment size is different from tiles size
    if augment_size != tiles_size:
        transform_list.insert(0, A.Resize(augment_size, augment_size))

    transform = A.Compose(transform_list)

    return transform
