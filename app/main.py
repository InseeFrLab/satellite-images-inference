"""
Main file for the API.
"""

import os
import mlflow
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box


from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, Query, Response
from app.logger_config import configure_logger
from app.utils import (
    get_file_system,
    get_model,
    get_normalization_metrics,
    create_geojson_from_mask,
    predict,
    predict_roi,
    get_filename_to_polygons,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the lifespan of the API.
    This context manager is used to load the ML model and other resources
    when the API starts and clean them up when the API stops.
    Args:
        app (FastAPI): The FastAPI application.
    """
    global \
        logger, \
        model, \
        n_bands, \
        tiles_size, \
        augment_size, \
        module_name, \
        normalization_mean, \
        normalization_std

    gdal.UseExceptions()
    logger = configure_logger()

    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    # Load the ML model
    model = get_model(model_name, model_version)

    # Extract several variables from model metadata
    n_bands = int(mlflow.get_run(model.metadata.run_id).data.params["n_bands"])
    tiles_size = int(mlflow.get_run(model.metadata.run_id).data.params["tiles_size"])
    augment_size = int(mlflow.get_run(model.metadata.run_id).data.params["augment_size"])
    module_name = mlflow.get_run(model.metadata.run_id).data.params["module_name"]
    normalization_mean, normalization_std = get_normalization_metrics(model, n_bands)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Satellite Image Inference",
    description="Segment satellite images",
    version="0.0.1",
)


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with current model name and version.
    """
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    return {
        "message": "Satellite Image Inference",
        "model_name": f"{model_name}",
        "model_version": f"{model_version}",
    }


@app.get("/predict_image", tags=["Predict Image"])
async def predict_image(image: str, polygons: bool = False) -> Dict:
    """
    Predicts mask for a given satellite image.

    Args:
        image (str): Path to the satellite image.
        polygons (bool, optional): Flag indicating whether to include polygons in the response. Defaults to False.

    Returns:
        Dict: Response containing the mask of the prediction.

    Raises:
        ValueError: If the dimension of the image is not divisible by the tile size used during training or if the dimension is smaller than the tile size.

    """
    logger.info(f"Predict image endpoint accessed with image: {image}")

    lsi = predict(
        image=image,
        model=model,
        tiles_size=tiles_size,
        augment_size=augment_size,
        n_bands=n_bands,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        module_name=module_name,
    )
    if polygons:
        return Response(content=create_geojson_from_mask(lsi).to_json(), media_type="text/plain")
    else:
        return {"mask": lsi.label.tolist()}


@app.get("/predict_cluster", tags=["Predict Cluster"])
def predict_cluster(
    cluster_id: str,
    year: int = Query(2022, ge=2017, le=2023),
    dep: str = Query("MAYOTTE", regex="^(MAYOTTE|GUADELOUPE|MARTINIQUE|GUYANE|REUNION)$"),
) -> Dict:
    """
    Predicts cluster for a given cluster ID, year, and department.

    Args:
        cluster_id (str): The ID of the cluster.
        year (int): The year of the satellite images.
        dep (str): The department of the satellite images.

    Returns:
        Dict: Response containing the predicted cluster.
    """
    logger.info(
        f"Predict cluster endpoint accessed with cluster_id: {cluster_id}, year: {year}, and department: {dep}"
    )

    fs = get_file_system()

    with fs.open("projet-slums-detection/ilots/ilots.gpkg", "rb") as f:
        clusters = gpd.read_file(f)

    # Get the filename to polygons mapping
    filename_table = get_filename_to_polygons(dep, year, fs)

    # Get the selected cluster
    selected_cluster = clusters.loc[clusters["ident_ilot"] == cluster_id].to_crs(filename_table.crs)

    # Get the filenames of the images that intersect with the selected cluster
    images = filename_table.loc[
        filename_table.geometry.intersects(selected_cluster.geometry.iloc[0]),
        "filename",
    ].tolist()

    # Predict the cluster
    preds_cluster = predict_roi(
        images,
        selected_cluster,
        model,
        tiles_size,
        augment_size,
        n_bands,
        normalization_mean,
        normalization_std,
        module_name,
    )

    return Response(content=preds_cluster.loc[:, "geometry"].to_json(), media_type="text/plain")


@app.get("/predict_bbox", tags=["Predict Bounding Box"])
def predict_bbox(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    epsg: int = Query(4326, ge=0),
    year: int = Query(2022, ge=2017, le=2023),
    dep: str = Query("MAYOTTE", regex="^(MAYOTTE|GUADELOUPE|MARTINIQUE|GUYANE|REUNION)$"),
) -> Dict:
    """
    Predicts the bounding box for satellite images based on the given coordinates.

    Args:
        xmin (float): The minimum x-coordinate of the bounding box.
        xmax (float): The maximum x-coordinate of the bounding box.
        ymin (float): The minimum y-coordinate of the bounding box.
        ymax (float): The maximum y-coordinate of the bounding box.
        epsg (int, optional): The EPSG code of the coordinate reference system (CRS) for the bounding box. Defaults to 4326.
        year (int, optional): The year of the satellite images to use for prediction. Defaults to 2022.

    Returns:
        Dict: A dictionary containing the predicted bounding box coordinates.
    """
    logger.info(
        f"Predict bbox endpoint accessed with bounding box coordinates: ({xmin}, {xmax}, {ymin}, {ymax}), epsg: {epsg}, year: {year}, and department: {dep}"
    )

    fs = get_file_system()

    # Get the filename to polygons mapping
    filename_table = get_filename_to_polygons(dep, year, fs)

    # Create a GeoSeries with the bounding box
    bbox_geo = gpd.GeoSeries(box(*[xmin, ymin, xmax, ymax]), crs=epsg).to_crs(filename_table.crs)

    # Get the filenames of the images that intersect with the bbox
    images = filename_table.loc[
        filename_table.geometry.intersects(bbox_geo.geometry.iloc[0]),
        "filename",
    ].tolist()

    # Predict the bbox
    preds_bbox = predict_roi(
        images,
        bbox_geo,
        model,
        tiles_size,
        augment_size,
        n_bands,
        normalization_mean,
        normalization_std,
        module_name,
    )

    return Response(content=preds_bbox.loc[:, "geometry"].to_json(), media_type="text/plain")
