import argparse
import asyncio
from datetime import datetime

import aiohttp
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm.asyncio import tqdm

from app.utils import get_file_system


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

    # Get cluster list
    clusters = (
        pq.ParquetDataset(
            "projet-slums-detection/data-clusters",
            filesystem=fs,
            filters=[("dep", "=", dep)],
        )
        .read(columns=["ident_ilot"])
        .column("ident_ilot")
        .to_pylist()
    )

    # On retire des clusters de Guyane très gros pour le moment
    clusters_to_ignore = [
        "97301PED1",
        "97301PED2",
        "97303PED3",
        "97303PED4",
        "97304PED7",
        "97306PED4",
        "97308PE10",
        "97308PED9",
        "97311PED3",
        "97311PED4",
        "97311PED8",
        "97312PED9",
        "97312PE11",
        "97314PED2",
        "97314PED3",
        "97352PED2",
        "97353PE11",
        "97356PED7",
        "97357PE36",
        "97357PE38",
        "97357PE40",
        "97358PED1",
        "97358PED2",
        "97360PE29",
        "97360PE30",
        "97360PE33",
        "97360PE31",
        "97360PE32",
        "97362PE12",
        "97362PE13",
    ]
    clusters = list(set(clusters) - set(clusters_to_ignore))

    urls = ["https://satellite-images-inference.lab.sspcloud.fr/predict_cluster"] * len(clusters)
    timeout = aiohttp.ClientTimeout(total=60 * 60 * 10)  # 10 heures timeout

    # Create an asynchronous HTTP client session
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            fetch(session, url, params={"year": f"{year}", "dep": f"{dep}", "cluster_id": cluster})
            for url, cluster in zip(urls, clusters)
        ]
        responses = await tqdm.gather(*tasks)

    # Create a dictionary mapping clusters to their corresponding predictions
    result = {k: v for k, v in zip(clusters, responses)}

    failed_query = []
    for cluster_id, pred in result.items():
        try:
            # Read the prediction file as a GeoDataFrame
            result[cluster_id] = gpd.read_file(pred["statistics"])
        except Exception as e:
            print(f"Error with cluster {cluster_id}: {str(e)}")
            print(f"Prediction returned: {pred}")
            # Get the list of failed clusters
            failed_query.append(cluster_id)

    # Set the maximum number of retries for failed clusters
    max_retry = 50
    counter = 0

    # Retry failed clusters up to the maximum number of retries
    while failed_query and counter < max_retry:
        urls = ["https://satellite-images-inference.lab.sspcloud.fr/predict_cluster"] * len(failed_query)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                fetch(session, url, params={"year": f"{year}", "dep": f"{dep}", "cluster_id": cluster})
                for url, cluster in zip(urls, failed_query)
            ]
            responses_retry = await tqdm.gather(*tasks)

            result_retry = {k: v for k, v in zip(failed_query, responses_retry)}

            failed_query = []
            for cluster_id, pred in result_retry.items():
                try:
                    # Update the result dictionary with the retry results for successful clusters
                    result[cluster_id] = gpd.read_file(pred["statistics"])
                except Exception as e:
                    print(f"Error with image {cluster_id}: {str(e)}")
                    # Get the list of failed clusters
                    failed_query.append(cluster_id)

        counter += 1

    # Filter out clusters with failed predictions from the result dictionary
    statistics = pd.concat([gdf for gdf in result.values() if isinstance(gdf, gpd.GeoDataFrame)])
    statistics = statistics[~statistics["geometry"].is_empty]

    # Saving the results
    statistics_path = f"""projet-slums-detection/data-prediction/PLEIADES/{dep}/{year}/{model_info["model_name"]}/{model_info["model_version"]}/statistics_clusters"""
    statistics.to_parquet(f"{statistics_path}.parquet", filesystem=fs)

    print(f"{failed_query}")


if __name__ == "__main__":
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Make predictions on a given department and year")

    parser.add_argument(
        "--year",
        type=int,
        choices=range(2017, datetime.now().year + 2),
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
