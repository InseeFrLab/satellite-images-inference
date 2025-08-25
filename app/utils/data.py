import json
import os
from typing import Dict

import geopandas as gpd
import mlflow
import numpy as np
import pyarrow.parquet as pq
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage
from s3fs import S3FileSystem


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


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=None,
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
        model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
        return model
    except Exception as error:
        raise Exception(f"Failed to fetch model {model_name} version {model_version}: {str(error)}") from error


def get_normalization_metrics(model_params: Dict) -> tuple:
    """
    Retrieves normalization metrics (mean and standard deviation) for the model.

    Args:
        model_params (Dict): A dictionary containing model parameters.
    Returns:
        Tuple: A tuple containing normalization mean and standard deviation.
    """
    normalization_mean = json.loads(model_params["normalization_mean"])
    normalization_std = json.loads(model_params["normalization_std"])
    n_bands = int(model_params["n_bands"])

    # Extract normalization mean and standard deviation for the number of bands
    normalization_mean, normalization_std = (
        normalization_mean[:n_bands],
        normalization_std[:n_bands],
    )

    return (normalization_mean, normalization_std)


def get_cache_path(image: str) -> str:
    """
    Get the cache path for the image.

    Args:
        image (str): The image path.

    Returns:
        str: The cache path.
    """
    assert "MLFLOW_MODEL_NAME" in os.environ, "Please set the MLFLOW_MODEL_NAME environment variable."
    assert "MLFLOW_MODEL_VERSION" in os.environ, "Please set the MLFLOW_MODEL_VERSION environment variable."

    cache_path = os.path.dirname(image.replace(image.split("/")[1], "cache-predictions"))
    image_name = os.path.splitext(os.path.basename(image))[0]
    return f"{cache_path}/{os.getenv('MLFLOW_MODEL_NAME')}/{os.getenv('MLFLOW_MODEL_VERSION')}/{image_name}.npy"


def load_from_cache(image: str, n_bands: list[int], filesystem: S3FileSystem) -> SegmentationLabeledSatelliteImage:
    """
    Load the image and mask from the cache.

    Args:
        image (str): The image path.
        n_bands (list[int]): The number of bands.
        filesystem (s3fs.S3FileSystem): The file system.

    Returns:
        SegmentationLabeledSatelliteImage: The labeled satellite image with the predicted mask.
    """

    mask_path = get_cache_path(image)

    si = get_satellite_image(image, n_bands)
    with filesystem.open(mask_path, "rb") as f:
        mask = np.load(f)

    logits = True if len(mask.shape) >= 3 else False
    return SegmentationLabeledSatelliteImage(si, mask, logits=logits)


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
        pq.ParquetDataset(
            "projet-slums-detection/data-raw/PLEIADES/filename-to-polygons/",
            filesystem=fs,
            filters=[("dep", "=", dep), ("year", "=", year)],
        )
        .read()
        .to_pandas()
    )

    # Convert the geometry column to a GeoSeries
    data["geometry"] = gpd.GeoSeries.from_wkt(data["geometry"])
    return gpd.GeoDataFrame(data, geometry="geometry", crs=data.CRS.unique()[0])
