"""
Utils.
"""
import mlflow
import yaml
from astrovision.data import SatelliteImage
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch


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


def get_normalization_metrics(model: mlflow.pyfunc.PyFuncModel):
    n_bands = int(mlflow.get_run(model.metadata.run_id).data.params["n_bands"])

    params = yaml.safe_load(
        mlflow.artifacts.load_text(
            f"{mlflow.get_run(model.metadata.run_id).info.artifact_uri}/metrics-normalization.yaml"
        )
    )
    normalization_mean, normalization_std = params["mean"], params["std"]
    normalization_mean, normalization_std = (
        normalization_mean[:n_bands],
        normalization_std[:n_bands],
    )
    return (normalization_mean, normalization_std)


def get_satellite_image(model: mlflow.pyfunc.PyFuncModel, image_path: str):
    # TODO Raise error if n_bands different
    n_bands = int(mlflow.get_run(model.metadata.run_id).data.params["n_bands"])

    si = SatelliteImage.from_raster(
        file_path=f"/vsis3/{image_path}",
        dep=None,
        date=None,
        n_bands=n_bands,
    )
    return si


def get_transform(model: mlflow.pyfunc.PyFuncModel):
    tiles_size = int(mlflow.get_run(model.metadata.run_id).data.params["tiles_size"])
    augment_size = int(mlflow.get_run(model.metadata.run_id).data.params["augment_size"])

    normalization_mean, normalization_std = get_normalization_metrics(model)

    transform_list = [
        A.Normalize(
            max_pixel_value=1.0,
            mean=normalization_mean,
            std=normalization_std,
        ),
        ToTensorV2(),
    ]
    if augment_size != tiles_size:
        transform_list.insert(0, A.Resize(augment_size, augment_size))
    transform = A.Compose(transform_list)

    return transform


def preprocess_image(model: mlflow.pyfunc.PyFuncModel, image: SatelliteImage):
    transform = get_transform(model)
    normalized_si = transform(image=np.transpose(image.array, [1, 2, 0]))["image"].unsqueeze(dim=0)

    return normalized_si


def produce_mask(prediction: np.array, model: mlflow.pyfunc.PyFuncModel, image_size: tuple):
    module_name = mlflow.get_run(model.metadata.run_id).data.params["module_name"]

    match module_name:
        case "deeplabv3":
            mask = prediction.softmax(dim=1) > 0.5

        case "single_class_deeplabv3":
            mask = prediction.sigmoid() > 0.5

        case "segformer-b5":
            interpolated_prediction = torch.nn.functional.interpolate(
                prediction,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )

            mask = interpolated_prediction.sigmoid() > 0.5

        case _:
            raise ValueError("Unknown pattern")

    return mask.numpy()
