"""
Main file for the API.
"""

import os
import mlflow
from astrovision.plot import make_mosaic
from osgeo import gdal
from joblib import Parallel, delayed

from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI
from app.utils import (
    get_model,
    get_normalization_metrics,
    get_satellite_image,
    create_geojson_from_mask,
    make_prediction,
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


@app.get("/predict", tags=["Predict"])
async def predict(image: str, polygons: bool = False) -> Dict:
    """
    Predicts mask for a given satellite image.

    Args:
        request (PredictionRequest): Request containing image path and polygons flag.

    Returns:
        Dict: Response containing mask of prediction.
    """

    # Retrieve satellite image
    si = get_satellite_image(image, n_bands)

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

            lsi_splitted = Parallel(n_jobs=16)(
                delayed(make_prediction)(
                    s_si,
                    model,
                    tiles_size,
                    augment_size,
                    n_bands,
                    normalization_mean,
                    normalization_std,
                    module_name,
                )
                for s_si in si_splitted
            )

            lsi = make_mosaic(lsi_splitted, [i for i in range(n_bands)])
    else:
        raise ValueError(
            "The dimension of the image should be equal to or greater than the tile size used during training."
        )

    if polygons:
        return create_geojson_from_mask(lsi.label.astype("uint8"), si)
    else:
        return {"mask": lsi.label.tolist()}
