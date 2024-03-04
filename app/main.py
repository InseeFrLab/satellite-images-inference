"""
Main file for the API.
"""

import os
import mlflow
import numpy as np
import torch

from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI
from app.utils import (
    get_model,
    get_normalization_metrics,
    get_satellite_image,
    preprocess_image,
    produce_mask,
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
async def predict(
    image: str,
) -> Dict:
    """
    Predicts mask for a given satellite image.

    Args:
        image (str): S3 path of the satellite image.

    Returns:
        Dict: Response containing mask of prediction.
    """
    # Retrieve satellite image
    si = get_satellite_image(image, n_bands)

    if si.array.shape[1] == tiles_size:
        # Preprocess the image
        normalized_si = preprocess_image(
            model=model,
            image=si,
            tiles_size=tiles_size,
            augment_size=augment_size,
            n_bands=n_bands,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

        # Make prediction using the model
        prediction = torch.tensor(model.predict(normalized_si.numpy()))

        # Produce mask from prediction
        mask = produce_mask(prediction, model, module_name, si.array.shape[-2:])

    elif si.array.shape[1] > tiles_size:
        if si.array.shape[1] % tiles_size != 0:
            raise ValueError(
                "The dimension of the image must be divisible by the tiles size used during training."
            )
        else:
            si_splitted = si.split(tiles_size)
            masks = []
            for s_si in si_splitted:
                # Preprocess the image
                normalized_si = preprocess_image(
                    model=model,
                    image=s_si,
                    tiles_size=tiles_size,
                    augment_size=augment_size,
                    n_bands=n_bands,
                )

                # Make prediction using the model
                prediction = torch.tensor(model.predict(normalized_si.numpy()))

                # Produce mask from prediction
                masks.append(produce_mask(prediction, model, module_name, s_si.array.shape[-2:]))

            quotient = si.array.shape[1] / tiles_size
            grid = np.array(masks).reshape(quotient, quotient, tiles_size, tiles_size)

            # Stack the small arrays to create the large array
            mask = np.block([[grid[i, j] for j in range(quotient)] for i in range(quotient)])
    else:
        raise ValueError(
            "The dimension of the image should be equal to or greater than the tile size used during training."
        )

    # Convert mask to list and return as a dictionnary
    return {"mask": mask.tolist()}
    # arr = np.asarray(json.loads(resp.json()))  # resp.json() if using Python requests
