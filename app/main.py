"""
Main file for the API.
"""

import gc
import os
from contextlib import asynccontextmanager
from typing import Dict

import geopandas as gpd
import mlflow
import numpy as np
import pyarrow.parquet as pq
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from osgeo import gdal
from shapely.geometry import box

from app.logger_config import configure_logger
from app.utils import (
    compute_roi_statistics,
    create_geojson_from_mask,
    get_cache_path,
    get_file_system,
    get_filename_to_polygons,
    get_model,
    get_normalization_metrics,
    load_from_cache,
    predict,
    produce_mask,
    subset_predictions,
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
    gc.collect()

    fs = get_file_system()

    if not fs.exists(get_cache_path(image)):
        lsi = predict(
            images=image,
            model=model,
            tiles_size=tiles_size,
            augment_size=augment_size,
            n_bands=n_bands,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            module_name=module_name,
        )
        # Save predictions to cache
        with fs.open(get_cache_path(image), "wb") as f:
            np.save(f, lsi.label)

    else:
        logger.info(f"Loading prediction from cache for image: {image}")
        lsi = load_from_cache(image, n_bands, fs)

    # Produce mask with class IDs
    lsi.label = produce_mask(lsi.label, module_name)

    if polygons:
        return JSONResponse(content=create_geojson_from_mask(lsi).to_json())
    else:
        return {"mask": lsi.label.tolist()}


@app.get("/predict_cluster", tags=["Predict Cluster"])
def predict_cluster(
    cluster_id: str,
    year: int = Query(2022, ge=2017, le=2025),
    dep: str = Query(
        "MAYOTTE", regex="^(MAYOTTE|GUADELOUPE|MARTINIQUE|GUYANE|REUNION|SAINT-MARTIN)$"
    ),
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

    # Get cluster file
    clusters = (
        pq.ParquetDataset(
            "projet-slums-detection/data-clusters", filesystem=fs, filters=[("dep", "=", dep)]
        )
        .read()
        .to_pandas()
    )
    clusters["geometry"] = gpd.GeoSeries.from_wkt(clusters["geometry"])
    clusters = gpd.GeoDataFrame(clusters, geometry="geometry", crs="EPSG:4326")

    # Get the filename to polygons mapping
    filename_table = get_filename_to_polygons(dep, year, fs)

    # Get the selected cluster
    selected_cluster = clusters.loc[clusters["ident_ilot"] == cluster_id].to_crs(filename_table.crs)

    # Get the filenames of the images that intersect with the selected cluster
    images = filename_table.loc[
        filename_table.geometry.intersects(selected_cluster.geometry.iloc[0]),
        "filename",
    ].tolist()

    # Check if images are found in S3 bucket
    if not images:
        logger.info(
            f"""No images found for cluster_id: {cluster_id}, year: {year}, and department: {dep}"""
        )
        return JSONResponse(
            content={
                "predictions": gpd.GeoDataFrame(
                    columns=["geometry"], crs=filename_table.crs
                ).to_json(),
                "statistics": gpd.GeoDataFrame(
                    columns=["geometry"], crs=filename_table.crs
                ).to_json(),
            }
        )

    images_to_predict = [im for im in images if not fs.exists(get_cache_path(im))]
    images_from_cache = [im for im in images if fs.exists(get_cache_path(im))]
    predictions = []

    if images_to_predict:
        # Predict
        predictions = predict(
            images_to_predict,
            model,
            tiles_size,
            augment_size,
            n_bands,
            normalization_mean,
            normalization_std,
            module_name,
        )

        # Save predictions to cache
        for im, pred in zip(images_to_predict, predictions):
            with fs.open(get_cache_path(im), "wb") as f:
                np.save(f, pred.label)

    if images_from_cache:
        logger.info(
            f"""Loading predictions from cache for images: {", ".join(images_from_cache)}"""
        )
        # Load from cache
        predictions += [load_from_cache(im, n_bands, fs) for im in images_from_cache]

    # Produce mask with class IDs TODO : check if ok
    for lsi in predictions:
        lsi.label = produce_mask(lsi.label, module_name)

    # Restrict predictions to the selected cluster
    preds_cluster = subset_predictions(predictions, selected_cluster)

    stats_cluster = compute_roi_statistics(predictions, selected_cluster)

    response_data = {
        "predictions": preds_cluster.to_json(),
        "statistics": stats_cluster.to_json(),
    }

    return JSONResponse(content=response_data)


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

    # Check if images are found in S3 bucket
    if not images:
        logger.info(
            f"""No images found for bounding box: ({xmin}, {xmax}, {ymin}, {ymax}), epsg: {epsg}, year: {year}, and department: {dep}"""
        )
        return JSONResponse(
            content={
                "predictions": gpd.GeoDataFrame(
                    columns=["geometry"], crs=filename_table.crs
                ).to_json(),
                "statistics": gpd.GeoDataFrame(
                    columns=["geometry"], crs=filename_table.crs
                ).to_json(),
            }
        )

    images_to_predict = [im for im in images if not fs.exists(get_cache_path(im))]
    images_from_cache = [im for im in images if fs.exists(get_cache_path(im))]
    predictions = []

    if images_to_predict:
        # Predict the bbox
        predictions = predict(
            images_to_predict,
            model,
            tiles_size,
            augment_size,
            n_bands,
            normalization_mean,
            normalization_std,
            module_name,
        )

        # Save predictions to cache
        for im, pred in zip(images_to_predict, predictions):
            with fs.open(get_cache_path(im), "wb") as f:
                np.save(f, pred.label)

    if images_from_cache:
        logger.info(f"Loading predictions from cache for images: {", ".join(images_from_cache)}")
        # Load from cache
        predictions += [load_from_cache(im, n_bands, fs) for im in images_from_cache]

    # Produce mask with class IDs TODO : check if ok
    for lsi in predictions:
        lsi.label = produce_mask(lsi.label, module_name)

    preds_bbox = subset_predictions(predictions, bbox_geo)

    stats_bbox = compute_roi_statistics(predictions, bbox_geo)

    response_data = {
        "predictions": preds_bbox.to_json(),
        "statistics": stats_bbox.to_json(),
    }

    return JSONResponse(content=response_data)
