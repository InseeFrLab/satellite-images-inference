import geopandas as gpd
import mlflow
import pyarrow.parquet as pq
from s3fs import S3FileSystem

from app.utils import (
    get_normalization_metrics,
)


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


def get_model_from_id(run_id: str) -> mlflow.pyfunc.PyFuncModel:
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
        model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/model")
        return model
    except Exception as error:
        raise Exception(f"Failed to fetch model from run_id : {run_id}") from error


def fetch_model(run_id):
    # Load the ML model
    model = get_model_from_id(run_id)

    # Extract several variables from model metadata
    n_bands = int(mlflow.get_run(model.metadata.run_id).data.params["n_bands"])
    tiles_size = int(mlflow.get_run(model.metadata.run_id).data.params["tiles_size"])
    augment_size = int(mlflow.get_run(model.metadata.run_id).data.params["augment_size"])
    module_name = mlflow.get_run(model.metadata.run_id).data.params["module_name"]
    normalization_mean, normalization_std = get_normalization_metrics(model, n_bands)

    return {
        "model": model,
        "n_bands": n_bands,
        "tiles_size": tiles_size,
        "augment_size": augment_size,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "module_name": module_name,
    }
