"""
Utils.
"""

import os
import tempfile
from contextlib import contextmanager
from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from astrovision.data import SegmentationLabeledSatelliteImage
from rasterio.features import rasterize, shapes
from shapely import make_valid
from shapely.ops import unary_union

from app.logger_config import configure_logger

logger = configure_logger()


def produce_mask(
    prediction: np.array,
    module_name: str,
):
    """
    Produces mask from prediction array based on the specified model.

    Args:
        prediction (np.array): Array containing the prediction.
        module_name (str): Name of the module used for training.

    Returns:
        np.array: Mask generated from prediction array.
    """

    # Make prediction torch tensor
    prediction = torch.from_numpy(prediction)

    # Determine mask generation based on module name
    match module_name:
        case "deeplabv3":
            mask = prediction.softmax(dim=1) > 0.5

        case "single_class_deeplabv3":
            mask = prediction.sigmoid() > 0.5

        case "segformer-b5":
            mask = torch.argmax(prediction, dim=0)

        case _:
            raise ValueError("Invalid module name specified.")

    return mask.numpy()


@contextmanager
def temporary_raster():
    """Context manager for handling temporary raster files safely."""
    temp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    try:
        temp.close()
        yield temp.name
    finally:
        try:
            os.unlink(temp.name)
        except OSError:
            pass


def create_geojson_from_mask(lsi: SegmentationLabeledSatelliteImage) -> str:
    """
    Creates a Geopandas from a binary or multiclass mask.
    Args:
        lsi: A SegmentationLabeledSatelliteImage.
    Returns:
        A Geopandas representing the clusters with their respective labels.
    """
    # Convert label to uint8
    lsi.label = lsi.label.astype("uint8")

    # Define the metadata for the raster image
    metadata = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": lsi.label.shape[1],
        "height": lsi.label.shape[0],
        "crs": lsi.satellite_image.crs,
        "transform": rasterio.transform.from_origin(
            lsi.satellite_image.bounds[0], lsi.satellite_image.bounds[3], 0.5, 0.5
        ),  # pixel size is 0.5m
    }

    # Use the context manager for temporary file handling
    with temporary_raster() as temp_tif:
        with rasterio.open(temp_tif, "w+", **metadata) as dst:
            dst.write(lsi.label, 1)

            # Process shapes within the same rasterio context
            results = [
                {"properties": {"label": int(v)}, "geometry": s}
                for i, (s, v) in enumerate(shapes(lsi.label, mask=None, transform=dst.transform))
                if v != 0  # Keep only the labels which are not 0
            ]

    # Create and return GeoDataFrame
    if results:
        return gpd.GeoDataFrame.from_features(results)
    else:
        return gpd.GeoDataFrame(columns=["geometry", "label"])


def subset_predictions(
    predictions: list[SegmentationLabeledSatelliteImage],
    roi: gpd.GeoDataFrame,
) -> Dict:
    # Get the predictions for all the images
    preds = pd.concat([create_geojson_from_mask(x) for x in predictions])
    preds.crs = roi.crs

    # Ensure the geometries are valid
    if not all([geom.is_valid for geom in roi.geometry]):
        roi["geometry"] = roi["geometry"].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

    if not all([geom.is_valid for geom in preds.geometry]):
        preds["geometry"] = preds["geometry"].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

    # Union of the roi geometries
    roi_union = unary_union(roi.geometry)

    # Initialize a dictionary to store the results
    results = []

    # TODO: Peut-etre qu'on veut loop sur toutes les geometries plutôt pour garder l'info indiv des geométries mais plus couteux de faire N intersections

    # Iterate over each label in the predictions
    for label in preds["label"].unique():
        # Subset the predictions for the current label
        preds_label = preds[preds["label"] == label]

        # Union of the prediction geometries for the current label
        preds_union = unary_union(preds_label.geometry)

        # Perform the intersection with validated geometries
        geom_preds_in_roi = roi_union.intersection(preds_union)

        # Restrict the predictions to the region of interest
        preds_roi = gpd.GeoDataFrame(
            {"geometry": geom_preds_in_roi, "label": [label]},
            crs=roi.crs,
        )

        # Store the result in the dictionary
        results.append(preds_roi.reset_index(drop=True))

    results = pd.concat(results)
    return results[~results["geometry"].is_empty]


def compute_roi_statistics(predictions: list, roi: gpd.GeoDataFrame) -> Dict[str, float]:
    """
    Compute statistics of the predictions within a region of interest (ROI).

    Args:
        predictions (list): List of predictions.
        roi (gpd.GeoDataFrame): Region of interest.

    Returns:
        dict: Dictionary containing the computed statistics.
    """
    RESOLUTION = 0.5
    area_cluster = 0
    area_building = 0

    for pred in predictions:
        polygon_mask = rasterize(
            [(roi.geometry.iloc[0], 1)],
            out_shape=pred.label.shape,
            transform=pred.satellite_image.transform,
            fill=0,
            dtype=np.uint8,
        )

        original_mask = pred.label
        area_cluster += (polygon_mask.sum() * RESOLUTION**2) / 1e6  # in km²
        # TODO: Assume 1 is label for buildings
        area_building += (((original_mask == 1) * polygon_mask).sum() * RESOLUTION**2) / 1e6  # in km²

    pct_building = area_building / area_cluster * 100
    roi = roi.assign(area_cluster=area_cluster, area_building=area_building, pct_building=pct_building)

    return roi.reset_index(drop=True)
