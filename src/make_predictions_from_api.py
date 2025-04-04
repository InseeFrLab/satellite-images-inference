import argparse
import asyncio
import os
import tempfile

import aiohttp
import geopandas as gpd
import pandas as pd
import requests
from tqdm.asyncio import tqdm

from app.utils import get_file_system

# from src.postprocessing.postprocessing import clean_prediction
from src.retrievals.wrappers import get_filename_to_polygons


def save_geopackage_to_s3(gdf, s3_path, filesystem):
    """
    Save a GeoDataFrame as a GeoPackage to S3.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame to save
    s3_path : str
        The S3 path where to save the file (including .gpkg extension)
    filesystem : s3fs.S3FileSystem
        Initialized S3 filesystem object
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        # Save to temporary file
        gdf.to_file(temp_path, driver="GPKG")

        # Upload to S3
        with open(temp_path, "rb") as file:
            with filesystem.open(s3_path, "wb") as s3_file:
                s3_file.write(file.read())
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def fetch(session, url, **kwargs):
    try:
        async with session.get(url, **kwargs) as response:
            response = await response.json()
            return response
    except asyncio.TimeoutError:
        print(f"Request timed out for URL: {url} and params: {kwargs}")
        return None
    except aiohttp.ClientPayloadError as e:
        print(f"ClientPayloadError for URL: {url}, Error: {e}, params: {kwargs}")
        return None
    except Exception as e:
        print(f"An error occurred for URL: {url}, Error: {e}, params: {kwargs}")
        return None


async def main(dep: str, year: int):
    """
    Perform satellite image inference and save the predictions.

    Args:
        dep (str): The department code.
        year (int): The year of the satellite images.
    """

    # Get info of the model
    model_info = requests.get("https://satellite-images-inference.lab.sspcloud.fr/").json()

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

    urls = ["https://satellite-images-inference.lab.sspcloud.fr/predict_image"] * len(images)
    timeout = aiohttp.ClientTimeout(total=60 * 60 * 10)  # 10 heures timeout

    # Create an asynchronous HTTP client session
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [fetch(session, url, params={"image": image, "polygons": "True"}) for url, image in zip(urls, images)]
        responses = await tqdm.gather(*tasks)

    # Create a dictionary mapping images to their corresponding predictions
    result = {k: v for k, v in zip(images, responses)}

    failed_query = []
    for im, pred in result.items():
        try:
            # Read the prediction file as a GeoDataFrame
            result[im] = gpd.read_file(pred)
            result[im]["filename"] = im
        except Exception as e:
            print(f"Error with image {im}: {str(e)}")
            print(f"Prediction returned: {pred}")
            # Get the list of failed images
            failed_query.append(im)

    # Set the maximum number of retries for failed images
    max_retry = 50
    counter = 0

    # Retry failed images up to the maximum number of retries
    while failed_query and counter < max_retry:
        urls = ["https://satellite-images-inference.lab.sspcloud.fr/predict_image"] * len(failed_query)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [fetch(session, url, params={"image": image, "polygons": "True"}) for url, image in zip(urls, failed_query)]
            responses_retry = await tqdm.gather(*tasks)

            result_retry = {k: v for k, v in zip(failed_query, responses_retry)}

            failed_query = []
            for im, pred in result_retry.items():
                try:
                    # Update the result dictionary with the retry results for successful images
                    result[im] = gpd.read_file(pred)
                    result[im]["filename"] = im
                except Exception as e:
                    print(f"Error with image {im}: {str(e)}")
                    # Get the list of failed images
                    failed_query.append(im)

        counter += 1

    # Filter out images with failed predictions from the result dictionary
    predictions = pd.concat([gdf for gdf in result.values() if isinstance(gdf, gpd.GeoDataFrame)])
    predictions.crs = roi.crs
    # predictions = clean_prediction(predictions, buffer_distance=3)
    predictions = predictions[~predictions["geometry"].is_empty]

    # Saving the results
    predictions_path = f"""projet-slums-detection/data-prediction/PLEIADES/{dep}/{year}/{model_info["model_name"]}/{model_info["model_version"]}/predictions"""
    predictions.to_parquet(f"{predictions_path}.parquet", filesystem=fs)
    save_geopackage_to_s3(
        predictions.loc[:, predictions.columns != "filename"],
        f"{predictions_path}.gpkg",
        filesystem=fs,
    )

    print(f"{failed_query}")


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Make predictions on a given department and year")

    parser.add_argument(
        "--year",
        type=int,
        choices=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        metavar="N",
        default=2020,
        help="Year of the dataset to make predictions on",
        required=True,
    )
    parser.add_argument(
        "--dep",
        type=str,
        choices=["MAYOTTE", "GUADELOUPE", "MARTINIQUE", "GUYANE", "REUNION", "SAINT-MARTIN"],
        default="MAYOTTE",
        help="Department to make predictions on",
        required=True,
    )

    args = parser.parse_args()

    args_dict = vars(args)
    asyncio.run(main(**args_dict))
