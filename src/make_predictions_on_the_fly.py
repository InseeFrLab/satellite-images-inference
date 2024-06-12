from typing import Union
import geopandas as gpd
from tqdm import tqdm
import pandas as pd
import argparse
from s3fs import S3FileSystem
import os
import pyarrow.dataset as ds
import mlflow
import networkx as nx
from geopandas import GeoSeries
from shapely.geometry import MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
import libpysal
from app.utils import (
    get_file_system,
    get_normalization_metrics,
    create_geojson_from_mask,
    predict,
)


def get_filename_to_polygons(dep: str, year: int, fs: S3FileSystem) -> gpd.GeoDataFrame:
    """
    Retrieves the filename to polygons mapping for a given department and year.

    Args:
        dep (str): The department code.
        year (int): The year.
        fs (S3FileSystem): The S3FileSystem object for accessing the data.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filename to polygons mapping.

    """
    # Load the filename to polygons mapping
    data = (
        ds.dataset(
            "projet-slums-detection/data-raw/PLEIADES/filename-to-polygons/",
            partitioning=["dep", "year"],
            format="parquet",
            filesystem=fs,
        )
        .to_table()
        .filter((ds.field("dep") == f"dep={dep}") & (ds.field("year") == f"year={year}"))
        .to_pandas()
    )

    # Convert the geometry column to a GeoSeries
    data["geometry"] = gpd.GeoSeries.from_wkt(data["geometry"])
    return gpd.GeoDataFrame(data, geometry="geometry", crs=data.loc[0, "CRS"])


def check_line_intersection(
    poly1: Union[LineString, MultiLineString], poly2: Union[LineString, MultiLineString]
) -> bool:
    """
    Check if two lines intersect.

    Args:
        poly1 (Polygon): The first line.
        poly2 (Polygon): The second line.

    Returns:
        bool: True if the lines intersect, False otherwise.
    """
    intersection = poly1.intersection(poly2)
    return isinstance(intersection, (LineString, MultiLineString))


def merge_adjacent_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge adjacent polygons in a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to process.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with adjacent polygons merged.
    """
    # TODO: potentially add tiny buffer on polygons ?
    # Create a spatial weights matrix
    W = libpysal.weights.Queen.from_dataframe(gdf)
    # Merge adjacent polygons
    components = W.component_labels
    merged_gdf = gdf.dissolve(by=components)
    return merged_gdf


def clean_prediction(gdf_original, buffer_distance=3):
    """
    Clean the predictions by merging adjacent polygons and removing small artifacts.

    Args:
        gdf_original (gpd.GeoDataFrame): The original GeoDataFrame containing the predictions.
        buffer_distance (float, optional): The buffer distance to apply to the polygons. Defaults to 1.5.

    Returns:
        gpd.GeoDataFrame: The cleaned GeoDataFrame.

    """
    gdf = merge_adjacent_polygons(gdf_original.copy())

    # Buffer the geometry
    gdf["geometry"] = gdf["geometry"].buffer(-buffer_distance).buffer(buffer_distance)
    sindex = gdf.sindex

    # Find touching pairs
    touching_pairs = set()
    for idx, poly in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Processing geometries"):
        neighbors_indices = set(sindex.query(poly.geometry, predicate="touches"))
        touching_pairs.update(
            (min(idx, neighbor_idx), max(idx, neighbor_idx))
            for neighbor_idx in neighbors_indices
            if neighbor_idx != idx
            and check_line_intersection(poly.geometry, gdf.iloc[neighbor_idx].geometry)
        )

    # Create a connected graph
    G = nx.Graph(touching_pairs)
    connected_components = list(nx.connected_components(G))

    gdf_new = gpd.GeoDataFrame({"filename": pd.Series([])}, geometry=GeoSeries())

    to_remove = set()
    for connected_indexes_poly in connected_components:
        new_poly = unary_union(gdf.loc[list(connected_indexes_poly), "geometry"])
        filenames = gdf.loc[list(connected_indexes_poly), "filename"].tolist()
        new_data = gpd.GeoDataFrame({"filename": filenames, "geometry": new_poly})
        if not isinstance(new_poly, MultiPolygon):
            gdf_new = pd.concat([gdf_new, new_data], ignore_index=True)
            to_remove.update(connected_indexes_poly)

    gdf = gdf.drop(index=list(to_remove))
    gdf["filename"] = gdf["filename"].apply(lambda x: [x])
    gdf = pd.concat([gdf, gdf_new], ignore_index=True)
    gdf = gdf[~gdf["geometry"].is_empty].reset_index(drop=True)

    # Buffer the geometry back to original size
    gdf["geometry"] = gdf["geometry"].buffer(-buffer_distance).buffer(buffer_distance)

    return gdf


def get_model(run_id: str) -> mlflow.pyfunc.PyFuncModel:
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
        model = mlflow.pyfunc.load_model(model_uri=f"runs:/{run_id}/model")
        return model
    except Exception as error:
        raise Exception(f"Failed to fetch model from run_id : {run_id}") from error


def fetch_model(run_id):
    # Load the ML model
    model = get_model(run_id)

    # Extract several variables from model metadata
    n_bands = int(mlflow.get_run(model.metadata.run_id).data.params["n_bands"])
    tiles_size = int(mlflow.get_run(model.metadata.run_id).data.params["tiles_size"])
    augment_size = int(mlflow.get_run(model.metadata.run_id).data.params["augment_size"])
    module_name = mlflow.get_run(model.metadata.run_id).data.params["module_name"]
    normalization_mean, normalization_std = get_normalization_metrics(model, n_bands)

    return {
        "model": model,
        "n_bands": n_bands,
        "tiles_size": tiles_size,
        "augment_size": augment_size,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "module_name": module_name,
    }


def main(dep: str, year: int):
    """
    Perform satellite image inference and save the predictions.

    Args:
        dep (str): The department code.
        year (int): The year of the satellite images.
    """

    run_id: str = os.getenv("MLFLOW_MODEL_RUN_ID")

    # Get info of the model
    model_info = fetch_model(run_id)
    # Get file system
    fs = get_file_system()

    # Get the filename to polygons mapping
    filename_table = get_filename_to_polygons(dep, year, fs)

    # Get Region of Interest
    roi = gpd.read_file(fs.open(f"projet-slums-detection/data-roi/{dep}.geojson", "rb"))

    # Restrict to ROI
    images = filename_table.loc[
        filename_table.geometry.intersects(roi.geometry.iloc[0]),
        "filename",
    ].tolist()

    failed_images = []
    predictions = []

    for im in tqdm(images):
        try:
            lsi = predict(
                image=im,
                model=model_info["model"],
                tiles_size=model_info["tiles_size"],
                augment_size=model_info["augment_size"],
                n_bands=model_info["n_bands"],
                normalization_mean=model_info["normalization_mean"],
                normalization_std=model_info["normalization_std"],
                module_name=model_info["module_name"],
            )

            predictions.append(create_geojson_from_mask(lsi))
        except Exception as e:
            print(f"Error with image {im}: {str(e)}")
            # Get the list of failed images
            failed_images.append(im)

    # Set the maximum number of retries for failed images
    max_retry = 50
    counter = 0

    # Retry failed images up to the maximum number of retries
    while failed_images and counter < max_retry:
        for im in tqdm(failed_images):
            try:
                lsi = predict(
                    image=im,
                    model=model_info["model"],
                    tiles_size=model_info["tiles_size"],
                    augment_size=model_info["augment_size"],
                    n_bands=model_info["n_bands"],
                    normalization_mean=model_info["normalization_mean"],
                    normalization_std=model_info["normalization_std"],
                    module_name=model_info["module_name"],
                )

                predictions.append(create_geojson_from_mask(lsi))
                failed_images.remove(im)
            except Exception as e:
                print(f"Error with image {im}: {str(e)}")
                # Get the list of failed images
        counter += 1

    # Filter out images with failed predictions from the result dictionary
    predictions = pd.concat([gdf for gdf in predictions if isinstance(gdf, gpd.GeoDataFrame)])
    predictions.crs = roi.crs
    predictions = clean_prediction(predictions, buffer_distance=3)
    predictions_path = f"""projet-slums-detection/data-prediction/PLEIADES/{dep}/{year}/RUN-ID/{run_id}/predictions"""
    predictions.to_parquet(f"{predictions_path}.parquet", filesystem=fs)

    with fs.open(f"{predictions_path}.gpkg", "wb") as file:
        predictions.to_file(file, driver="GPKG")

    print(f"{failed_images}")


if __name__ == "__main__":
    assert (
        "MLFLOW_MODEL_RUN_ID" in os.environ
    ), "Please set the MLFLOW_MODEL_RUN_ID environment variable."

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Make predictions on a given department and year")

    parser.add_argument(
        "--year",
        type=int,
        choices=[2017, 2018, 2019, 2020, 2021, 2022, 2023],
        metavar="N",
        default=2020,
        help="Year of the dataset to make predictions on",
        required=True,
    )
    parser.add_argument(
        "--dep",
        type=str,
        choices=["MAYOTTE", "GUADELOUPE", "MARTINIQUE", "GUYANE", "REUNION"],
        default="MAYOTTE",
        help="Department to make predictions on",
        required=True,
    )

    args = parser.parse_args()

    args_dict = vars(args)
    main(**args_dict)
