import argparse
import os

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from app.utils import (
    create_geojson_from_mask,
    get_file_system,
    predict,
)
from src.postprocessing.postprocessing import clean_prediction
from src.retrievals.wrappers import fetch_model, get_filename_to_polygons


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
    images = images[10:12] + [
        "projet-slums-detection/data-raw/PLEIADES/MAYOTTE_CLEAN/2022/ORT_976_2022_0513_8568_U38S_8Bits.tif",
        "projet-slums-detection/data-raw/PLEIADES/MAYOTTE_CLEAN/2022/ORT_976_2022_0513_8593_U38S_8Bits.tif",
    ]
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

            predictions.append(gpd.GeoDataFrame(create_geojson_from_mask(lsi)))
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

                predictions.append(gpd.GeoDataFrame(create_geojson_from_mask(lsi)))
                failed_images.remove(im)
            except Exception as e:
                print(f"Error with image {im}: {str(e)}")
                # Get the list of failed images
        counter += 1

    # Filter out images with failed predictions from the result dictionary
    predictions = pd.concat([gdf for gdf in predictions if isinstance(gdf, gpd.GeoDataFrame)])
    if predictions.empty:
        print(f"There is 0 prediction for this dataset: {dep}_{year}")
        return f"There is 0 prediction for this dataset: {dep}_{year}"

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
        choices=["MAYOTTE", "GUADELOUPE", "MARTINIQUE", "GUYANE", "REUNION", "MAYOTTE_CLEAN"],
        default="MAYOTTE",
        help="Department to make predictions on",
        required=True,
    )

    args = parser.parse_args()

    args_dict = vars(args)
    main(**args_dict)
