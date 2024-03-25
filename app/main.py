"""
Main file for the API.
"""

import os
import mlflow
from osgeo import gdal
import geopandas as gpd
from shapely.ops import unary_union
import pandas as pd
from joblib import Parallel, delayed

from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, Query, Response
from app.utils import (
    get_file_system,
    get_model,
    get_normalization_metrics,
    get_satellite_image,
    create_geojson_from_mask,
    predict,
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
        model, \
        n_bands, \
        tiles_size, \
        augment_size, \
        module_name, \
        normalization_mean, \
        normalization_std

    gdal.UseExceptions()
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
    year: int = Query(..., ge=2017, le=2023),
    dep: str = Query(..., regex="^(MAYOTTE|GUADELOUPE|MARTINIQUE|GUYANE|REUNION)$"),
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
    fs = get_file_system()

    with fs.open("projet-slums-detection/ilots/ilots.gpkg", "rb") as f:
        clusters = gpd.read_file(f)

    with fs.open(
        f"projet-slums-detection/data-raw/PLEIADES/{dep}/{year}/filename_to_polygon.parquet",
        "rb",
    ) as f:
        filename_table = gpd.read_parquet(f)

    # Get the selected cluster
    selected_cluster = clusters.loc[clusters["ident_ilot"] == cluster_id].to_crs(filename_table.crs)

    # Get the filenames of the images that intersect with the selected cluster
    images = filename_table.loc[
        filename_table.geometry.intersects(selected_cluster.geometry.iloc[0]),
        "filename",
    ].tolist()

    # Predict the images
    n_jobs = min(len(images), 10)
    predictions = Parallel(n_jobs=n_jobs)(
        delayed(predict)(
            image,
            model,
            tiles_size,
            augment_size,
            n_bands,
            normalization_mean,
            normalization_std,
            module_name,
        )
        for image in images
    )

    # Get the crs from the first image
    crs = get_satellite_image(images[0], n_bands).crs

    # Get the predictions for all the images
    all_preds = pd.concat([create_geojson_from_mask(x) for x in predictions])
    all_preds.crs = crs

    # Restrict the predictions to the cluster
    preds_ilot = gpd.GeoDataFrame(
        geometry=[
            unary_union(selected_cluster.geometry).intersection(unary_union(all_preds.geometry))
        ],
        crs=selected_cluster.crs,
    )

    return Response(content=preds_ilot.loc[:, "geometry"].to_json(), media_type="text/plain")
