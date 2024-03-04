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


def get_normalization_metrics(model: mlflow.pyfunc.PyFuncModel, n_bands: int):
    """
    Retrieves normalization metrics (mean and standard deviation) for the model.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        n_bands (int): Number of bands in the satellite image.

    Returns:
        Tuple: A tuple containing normalization mean and standard deviation.
    """
    # Load normalization parameters from metrics-normalization.yaml
    params = yaml.safe_load(
        mlflow.artifacts.load_text(
            f"{mlflow.get_run(model.metadata.run_id).info.artifact_uri}/metrics-normalization.yaml"
        )
    )

    # Extract normalization mean and standard deviation for the number of bands
    normalization_mean, normalization_std = params["mean"], params["std"]
    normalization_mean, normalization_std = (
        normalization_mean[:n_bands],
        normalization_std[:n_bands],
    )

    return (normalization_mean, normalization_std)


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


def get_transform(
    model: mlflow.pyfunc.PyFuncModel, tiles_size: int, augment_size: int, n_bands: int
):
    """
    Retrieves the transformation pipeline for image preprocessing.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        tiles_size (int): Size of the satellite image.
        augment_size (int): .

    Returns:
        albumentations.Compose: A composition of image transformations.
    """
    # Retrieve normalization metrics
    normalization_mean, normalization_std = get_normalization_metrics(model, n_bands)

    # Define the transformation pipeline
    transform_list = [
        A.Normalize(
            max_pixel_value=1.0,
            mean=normalization_mean,
            std=normalization_std,
        ),
        ToTensorV2(),
    ]

    # Add resizing transformation if augment size is different from tiles size
    if augment_size != tiles_size:
        transform_list.insert(0, A.Resize(augment_size, augment_size))

    transform = A.Compose(transform_list)

    return transform


def preprocess_image(
    model: mlflow.pyfunc.PyFuncModel,
    image: SatelliteImage,
    tiles_size: int,
    augment_size: int,
    n_bands: int,
):
    """
    Preprocesses a satellite image using the specified model.

    Args:
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        image (SatelliteImage): SatelliteImage object representing the input image.
        tiles_size (int): Size of the satellite image.
        augment_size (int): .
    Returns:
        torch.Tensor: Normalized and preprocessed image tensor.
    """
    # Obtain transformation pipeline
    transform = get_transform(model, tiles_size, augment_size, n_bands)

    # Apply transformation to image
    normalized_si = transform(image=np.transpose(image.array, [1, 2, 0]))["image"].unsqueeze(dim=0)

    return normalized_si


def produce_mask(
    prediction: np.array, model: mlflow.pyfunc.PyFuncModel, module_name: str, image_size: tuple
):
    """
    Produces mask from prediction array based on the specified model.

    Args:
        prediction (np.array): Array containing the prediction.
        model (mlflow.pyfunc.PyFuncModel): MLflow PyFuncModel object representing the model.
        module_name (str): Name of the module used for training.
        image_size (tuple): Size of the original image.

    Returns:
        np.array: Mask generated from prediction array.
    """
    # Determine mask generation based on module name
    match module_name:
        case "deeplabv3":
            mask = prediction.softmax(dim=1) > 0.5

        case "single_class_deeplabv3":
            mask = prediction.sigmoid() > 0.5

        case "segformer-b5":
            # Interpolate prediction to original image size
            interpolated_prediction = torch.nn.functional.interpolate(
                prediction,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )
            mask = interpolated_prediction.sigmoid() > 0.5

        case _:
            raise ValueError("Invalid module name specified.")

    return mask.numpy()
