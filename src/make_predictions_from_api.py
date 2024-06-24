import geopandas as gpd
from tqdm.asyncio import tqdm
import pandas as pd
import argparse
import asyncio
import aiohttp
import requests
from src.postprocessing.postprocessing import clean_prediction
from src.retrievals.wrappers import get_filename_to_polygons
from app.utils import get_file_system


async def fetch(session, url, image):
    try:
        async with session.get(url, params={"image": image, "polygons": "True"}) as response:
            response_text = await response.text()
            return response_text
    except asyncio.TimeoutError:
        print(f"Request timed out for URL: {url} and image: {image}")
        return None
    except aiohttp.ClientPayloadError as e:
        print(f"ClientPayloadError for URL: {url}, Error: {e}, Image: {image}")
        return None
    except Exception as e:
        print(f"An error occurred for URL: {url}, Error: {e}, Image: {image}")
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
        tasks = [fetch(session, url, image) for url, image in zip(urls, images)]
        responses = await tqdm.gather(*tasks)

    # Create a dictionary mapping images to their corresponding predictions
    result = {k: v for k, v in zip(images, responses)}

    failed_images = []
    for im, pred in result.items():
        try:
            # Read the prediction file as a GeoDataFrame
            result[im] = gpd.read_file(pred, driver="GeoJSON")
            result[im]["filename"] = im
        except Exception as e:
            print(f"Error with image {im}: {str(e)}")
            print(f"Prediction returned: {pred}")
            # Get the list of failed images
            failed_images.append(im)

    # Set the maximum number of retries for failed images
    max_retry = 50
    counter = 0

    # Retry failed images up to the maximum number of retries
    while failed_images and counter < max_retry:
        urls = ["https://satellite-images-inference.lab.sspcloud.fr/predict_image"] * len(
            failed_images
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [fetch(session, url, image) for url, image in zip(urls, failed_images)]
            responses_retry = await tqdm.gather(*tasks)

            result_retry = {k: v for k, v in zip(failed_images, responses_retry)}

            failed_images = []
            for im, pred in result_retry.items():
                try:
                    # Update the result dictionary with the retry results for successful images
                    result[im] = gpd.read_file(pred, driver="GeoJSON")
                    result[im]["filename"] = im
                except Exception as e:
                    print(f"Error with image {im}: {str(e)}")
                    # Get the list of failed images
                    failed_images.append(im)

        counter += 1

    # Filter out images with failed predictions from the result dictionary
    predictions = pd.concat([gdf for gdf in result.values() if isinstance(gdf, gpd.GeoDataFrame)])
    predictions.crs = roi.crs
    predictions = clean_prediction(predictions, buffer_distance=3)
    predictions_path = f"""projet-slums-detection/data-prediction/PLEIADES/{dep}/{year}/{model_info["model_name"]}/{model_info["model_version"]}/predictions"""
    predictions.to_parquet(f"{predictions_path}.parquet", filesystem=fs)

    with fs.open(f"{predictions_path}.gpkg", "wb") as file:
        predictions.to_file(file, driver="GPKG")

    print(f"{failed_images}")


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Make predictions on a given department and year")

    parser.add_argument(
        "--year",
        type=int,
        choices=[2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
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
