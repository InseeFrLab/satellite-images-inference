"""
Main file for the API.
"""
import os
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI
import torch
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
    global model

    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    # Load the ML model
    model = get_model(model_name, model_version)
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


    Args:

        images (str): S3 path of an image

    Returns:

        Dict: Response containing mask of prediction.
    """
    si = get_satellite_image(model, image)

    normalized_si = preprocess_image(model=model, image=si)

    prediction = torch.tensor(model.predict(normalized_si.numpy()))

    mask = produce_mask(prediction, model, si.array.shape[-2:])
    return {"mask": mask.tolist()}
    # arr = np.asarray(json.loads(resp.json()))  # resp.json() if using Python requests
