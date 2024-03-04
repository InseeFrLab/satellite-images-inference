"""
Main file for the API.
"""
import os
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI
import torch
import mlflow
from app.utils import (
    get_model,
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
    global model, n_bands, tiles_size, augment_size, module_name

    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    # Load the ML model
    model = get_model(model_name, model_version)

    # Extract several variables from model metadata
    n_bands = int(mlflow.get_run(model.metadata.run_id).data.params["n_bands"])
    tiles_size = int(mlflow.get_run(model.metadata.run_id).data.params["tiles_size"])
    augment_size = int(mlflow.get_run(model.metadata.run_id).data.params["augment_size"])
    module_name = mlflow.get_run(model.metadata.run_id).data.params["module_name"]
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

    # Preprocess the image
    normalized_si = preprocess_image(
        model=model, image=si, tiles_size=tiles_size, augment_size=augment_size, n_bands=n_bands
    )

    # Make prediction using the model
    prediction = torch.tensor(model.predict(normalized_si.numpy()))

    # Produce mask from prediction
    mask = produce_mask(prediction, model, module_name, si.array.shape[-2:])

    # Convert mask to list and return as a dictionnary
    return {"mask": mask.tolist()}
    # arr = np.asarray(json.loads(resp.json()))  # resp.json() if using Python requests
