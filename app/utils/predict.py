from typing import List

import mlflow.pyfunc
import numpy as np
import torch
from astrovision.data import SegmentationLabeledSatelliteImage
from astrovision.plot import make_mosaic
from scipy.special import softmax
from tqdm import tqdm

from app.logger_config import configure_logger
from app.utils.data import get_cache_path, get_file_system, load_from_cache
from app.utils.preprocess_image import get_transform
from app.utils.split_and_normalize import get_normalized_sis

logger = configure_logger()


def predict(
    images: str | list[str],
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
    sliding_window_split: bool = False,
    overlap: int = None,
    batch_size: int = 25,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    """
    Predicts mask for a given satellite image or a given list of given satellite image.

    Args:
        images (str | list[str]): The path to the satellite image or a list of paths.
        model (mlflow.pyfunc.PyFuncModel): The MLflow PyFuncModel object representing the model.
        tiles_size (int): The size of the tiles used during training.
        augment_size (int): The size of the augmentation applied to the image.
        n_bands (int): The number of bands in the satellite image.
        normalization_mean (List[float]): The mean values for normalization.
        normalization_std (List[float]): The standard deviation values for normalization.
        sliding_window_split (bool, optional): Whether to use sliding window split. Defaults to False.
        overlap (int, optional): Overlap size for sliding window split. Defaults to None.
        batch_size (int, optional): The size of the batch for prediction. Defaults to 75.

    Returns:
        list[SegmentationLabeledSatelliteImage]: The labeled satellite image with the predicted mask.

    Raises:
        ValueError: If the dimension of the image is not divisible by the tile size used during training or if the dimension is smaller than the tile size.

    """

    if sliding_window_split and overlap is None:
        raise ValueError("If sliding_window_split is set to True, overlap must be specified.")
    elif not sliding_window_split and overlap is not None:
        logger.warning("sliding_window_split is set to False: overlap will be ignored.")

    all_predictions = []

    transform = get_transform(tiles_size, augment_size, n_bands, normalization_mean, normalization_std)
    fs = get_file_system()

    images = [images] if isinstance(images, str) else images

    model = model.to(device)
    for image in tqdm(images):
        # Check if the image is already cached
        cache_path = get_cache_path(image)
        if fs.exists(cache_path):
            logger.info(f"Loading {image} from cache.")
            all_predictions.append(load_from_cache(image, n_bands, fs))

        else:
            normalized_sis_tensor, si_splitted = get_normalized_sis(
                image, n_bands, tiles_size, normalization_mean, transform, sliding_window_split, overlap
            )  # tensor shape (num_tiles, n_bands, augment_size, augment_size) + list[SatelliteImage]

            prediction = make_batched_prediction(
                normalized_si=normalized_sis_tensor.to(device),
                model=model,
                tiles_size=tiles_size,
                batch_size=batch_size,
            )  # already softmaxed

            # Transform to labeled SI and make mosaic to get full prediction
            lsi_splitted = [
                SegmentationLabeledSatelliteImage(si_splitted[i], prediction[i], logits=True) for i in range(len(si_splitted))
            ]
            lsi = make_mosaic(lsi_splitted, [i for i in range(n_bands)])  # get back to full image
            all_predictions.append(lsi)

            # Save predictions to cache
            with fs.open(get_cache_path(image), "wb") as f:
                np.save(f, lsi.label)

    return all_predictions


def predict_batch(batch, model, tiles_size):
    prediction = model(batch)
    if prediction.shape[-2:] != (tiles_size, tiles_size):
        prediction = (
            torch.nn.functional.interpolate(
                prediction,
                size=tiles_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    return prediction


def make_batched_prediction(
    normalized_si: torch.Tensor,
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    batch_size: int = 25,
):
    """
    Makes a prediction on a satellite image.

    Args:
        normalized_si (torch.Tensor): The preprocessed and normalized satellite image tensor (output of preprocess_image).
        model: The ML model.
        tiles_size: The size of the tiles used during training.
        batch_size (int): The size of the batch for prediction.

    Returns:
        predictions_softmaxed (np.array): The predicted mask for the satellite image.
    """

    n_batches = len(normalized_si) // batch_size
    with torch.no_grad():
        predictions = []
        for i in tqdm(range(n_batches)):
            batch = normalized_si[batch_size * i : batch_size * (i + 1)]
            prediction = predict_batch(batch, model, tiles_size)
            predictions.append(prediction)

        if len(normalized_si) % batch_size != 0:
            batch = normalized_si[batch_size * n_batches :]
            prediction = predict_batch(batch, model, tiles_size)
            predictions.append(prediction)

    predictions = np.vstack(predictions)
    predictions_softmaxed = softmax(predictions, axis=1)

    return predictions_softmaxed
