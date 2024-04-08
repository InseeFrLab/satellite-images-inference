"""
Utils.
"""

import json
import os
from typing import Dict, List
import cv2

import albumentations as A
import geopandas as gpd
import mlflow
import numpy as np
import rasterio
import torch
from albumentations.pytorch.transforms import ToTensorV2
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage
from rasterio.features import shapes
from s3fs import S3FileSystem

from astrovision.plot import make_mosaic
from shapely.ops import unary_union
import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm
from app.logger_config import configure_logger

logger = configure_logger()


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def get_model(model_name: str, model_version: str) -> mlflow.pyfunc.PyFuncModel:
    """
    This function fetches a trained machine learning model from the MLflow
    model registry based on the specified model name and version.

    Args:
        model_name (str): The name of the model to fetch from the model
        registry.
        model_version (str): The version of the model to fetch from the model
        registry.
    Returns:
        model (mlflow.pyfunc.PyFuncModel): The loaded machine learning model.
    Raises:
        Exception: If the model fetching fails, an exception is raised with an
        error message.
    """

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        return model
    except Exception as error:
        raise Exception(
            f"Failed to fetch model {model_name} version {model_version}: {str(error)}"
        ) from error


def get_normalization_metrics(model: mlflow.pyfunc.PyFuncModel, n_bands: int):
    """
    Retrieves normalization metrics (mean and standard deviation) for the model.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        n_bands (int): Number of bands in the satellite image.

    Returns:
        Tuple: A tuple containing normalization mean and standard deviation.
    """
    normalization_mean = json.loads(
        mlflow.get_run(model.metadata.run_id).data.params["normalization_mean"]
    )
    normalization_std = json.loads(
        mlflow.get_run(model.metadata.run_id).data.params["normalization_std"]
    )

    # Extract normalization mean and standard deviation for the number of bands
    normalization_mean, normalization_std = (
        normalization_mean[:n_bands],
        normalization_std[:n_bands],
    )

    return (normalization_mean, normalization_std)


def get_satellite_image(image_path: str, n_bands: int):
    """
    Retrieves a satellite image specified by its path.

    Args:
        image_path (str): Path to the satellite image.
        n_bands (int): Number of bands in the satellite image.

    Returns:
        SatelliteImage: An object representing the satellite image.
    """

    # Load satellite image using the specified path and number of bands
    si = SatelliteImage.from_raster(
        file_path=f"/vsis3/{image_path}",
        dep=None,
        date=None,
        n_bands=n_bands,
    )
    return si


def get_transform(
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
):
    """
    Retrieves the transformation pipeline for image preprocessing.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
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


def preprocess_image(
    model: mlflow.pyfunc.PyFuncModel,
    image: SatelliteImage,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
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
    # Obtain transformation pipeline
    transform = get_transform(
        model, tiles_size, augment_size, n_bands, normalization_mean, normalization_std
    )

    # Deal when images to pred have more channels than images used during training
    if len(normalization_mean) != image.array.shape[0]:
        image.array = image.array[: len(normalization_mean)]

    # Apply transformation to image
    normalized_si = transform(image=np.transpose(image.array, [1, 2, 0]))["image"].unsqueeze(dim=0)

    return normalized_si


def produce_mask(
    prediction: np.array, model: mlflow.pyfunc.PyFuncModel, module_name: str, image_size: tuple
):
    """
    Produces mask from prediction array based on the specified model.

    Args:
        prediction (np.array): Array containing the prediction.
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        module_name (str): Name of the module used for training.
        image_size (tuple): Size of the original image.

    Returns:
        np.array: Mask generated from prediction array.
    """
    # Determine mask generation based on module name
    match module_name:
        case "deeplabv3":
            mask = prediction.softmax(dim=1) > 0.5

        case "single_class_deeplabv3":
            mask = prediction.sigmoid() > 0.5

        case "segformer-b5":
            # Interpolate prediction to original image size
            interpolated_prediction = torch.nn.functional.interpolate(
                prediction,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.argmax(interpolated_prediction, dim=1).squeeze()

        case _:
            raise ValueError("Invalid module name specified.")

    return mask.numpy()


def create_geojson_from_mask(lsi: SegmentationLabeledSatelliteImage) -> str:
    """
    Creates a GeoJSON string from a binary mask.

    Args:
        lsi: A SegmentationLabeledSatelliteImage.

    Returns:
        A GeoJSON string representing the clusters with value 1 in the binary mask.

    Raises:
        None.
    """
    lsi.label = lsi.label.astype("uint8")

    # Define the metadata for the raster image
    metadata = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": lsi.label.shape[1],
        "height": lsi.label.shape[0],
        "crs": lsi.satellite_image.crs,
        "transform": rasterio.transform.from_origin(
            lsi.satellite_image.bounds[0], lsi.satellite_image.bounds[3], 0.5, 0.5
        ),  # pixel size is 0.5m
    }

    # Write the binary array as a raster image
    with rasterio.open("temp.tif", "w+", **metadata) as dst:
        dst.write(lsi.label, 1)
        results = [
            {"properties": {"raster_val": v}, "geometry": s}
            for i, (s, v) in enumerate(shapes(lsi.label, mask=None, transform=dst.transform))
            if v == 1  # Keep only the clusters with value 1
        ]

    if results:
        return gpd.GeoDataFrame.from_features(results).loc[:, "geometry"]
    else:
        return gpd.GeoDataFrame(columns=["geometry"])


def make_prediction(
    image: SatelliteImage,
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
    module_name: str,
):
    """
    Makes a prediction on a satellite image.

    Args:
        image (SatelliteImage): The input satellite image.
        model: The ML model.
        tiles_size: The size of the tiles used during training.
        augment_size: The size of the augmentation used during training.
        n_bands: The number of bands in the satellite image.
        normalization_mean: The mean value used for normalization.
        normalization_std: The standard deviation used for normalization.

    Returns:
        SegmentationLabeledSatelliteImage: The labeled satellite image with the predicted mask.
    """
    # Preprocess the image
    normalized_si = preprocess_image(
        model=model,
        image=image,
        tiles_size=tiles_size,
        augment_size=augment_size,
        n_bands=n_bands,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )

    # Make prediction using the model
    with torch.no_grad():
        prediction = torch.tensor(model.predict(normalized_si.numpy()))

    # Produce mask from prediction
    mask = produce_mask(prediction, model, module_name, image.array.shape[-2:])
    return SegmentationLabeledSatelliteImage(image, mask)


def predict(
    image: str,
    model: mlflow.pyfunc.PyFuncModel,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
    normalization_mean: List[float],
    normalization_std: List[float],
    module_name: str,
) -> Dict:
    """
    Predicts mask for a given satellite image.

    Args:
        image (str): Path to the satellite image.
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        tiles_size (int): Size of the satellite image.
        augment_size (int): Size of the augmentation.
        n_bands (int): Number of bands in the satellite image.
        normalization_mean (List[float]): List of mean values for normalization.
        normalization_std (List[float]): List of standard deviation values for normalization.
        module_name (str): Name of the module used for training.

    Returns:
        SegmentationLabeledSatelliteImage: The labeled satellite image with the predicted mask.

    Raises:
        ValueError: If the dimension of the image is not divisible by the tile size used during training or if the dimension is smaller than the tile size.

    """
    # Retrieve satellite image
    si = get_satellite_image(image, n_bands)

    # Normalize image if it is not in uint8
    if si.array.dtype is not np.dtype("uint8"):
        si.array = cv2.normalize(si.array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if si.array.shape[1] == tiles_size:
        lsi = make_prediction(
            model=model,
            image=si,
            tiles_size=tiles_size,
            augment_size=augment_size,
            n_bands=n_bands,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            module_name=module_name,
        )

    elif si.array.shape[1] > tiles_size:
        if si.array.shape[1] % tiles_size != 0:
            raise ValueError(
                "The dimension of the image must be divisible by the tiles size used during training."
            )
        else:
            si_splitted = si.split(tiles_size)

            lsi_splitted = [
                make_prediction(
                    s_si,
                    model,
                    tiles_size,
                    augment_size,
                    n_bands,
                    normalization_mean,
                    normalization_std,
                    module_name,
                )
                for s_si in tqdm(si_splitted)
            ]

            lsi = make_mosaic(lsi_splitted, [i for i in range(n_bands)])
    else:
        raise ValueError(
            "The dimension of the image should be equal to or greater than the tile size used during training."
        )
    return lsi


def predict_roi(
    images: List[str],
    roi: gpd.GeoDataFrame,
    model,
    tiles_size,
    augment_size,
    n_bands,
    normalization_mean,
    normalization_std,
    module_name,
):
    # Predict the images
    predictions = [
        predict(
            image,
            model,
            tiles_size,
            augment_size,
            n_bands,
            normalization_mean,
            normalization_std,
            module_name,
        )
        for image in tqdm(images)
    ]

    # Get the crs from the first image
    crs = get_satellite_image(images[0], n_bands).crs

    # Get the predictions for all the images
    all_preds = pd.concat([create_geojson_from_mask(x) for x in predictions])
    all_preds.crs = crs

    # Restrict the predictions to the region of interest
    preds_roi = gpd.GeoDataFrame(
        geometry=[unary_union(roi.geometry).intersection(unary_union(all_preds.geometry))],
        crs=roi.crs,
    )
    return preds_roi


def get_filename_to_polygons(dep: str, year: int, fs: S3FileSystem) -> gpd.GeoDataFrame:
    """
    Retrieves the filename to polygons mapping for a given department and year.

    Args:
        dep (str): The department code.
        year (int): The year.
        fs (S3FileSystem): The S3FileSystem object for accessing the data.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filename to polygons mapping.

    """
    # Load the filename to polygons mapping
    data = (
        ds.dataset(
            "projet-slums-detection/data-raw/PLEIADES/filename-to-polygons/",
            partitioning=["dep", "year"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .filter((ds.field("dep") == f"dep={dep}") & (ds.field("year") == f"year={year}"))
        .to_pandas()
    )

    # Convert the geometry column to a GeoSeries
    data["geometry"] = gpd.GeoSeries.from_wkt(data["geometry"])
    return gpd.GeoDataFrame(data, geometry="geometry", crs=data.loc[0, "CRS"])
