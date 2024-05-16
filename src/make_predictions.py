import geopandas as gpd
from tqdm.asyncio import tqdm
import pandas as pd
import argparse
from s3fs import S3FileSystem
import os
import pyarrow.dataset as ds
import asyncio
import aiohttp
import requests
import networkx as nx
from geopandas import GeoSeries
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
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


def check_poly_intersection(poly1, poly2):
    intersection = poly1.intersection(poly2)
    if isinstance(intersection, (Polygon, MultiPolygon)):
        return True
    return False


def clean_prediction(gdf_original, buffer_distance=1.5):
    gdf = gdf_original.copy()

    gdf["geometry"] = gdf["geometry"].buffer(buffer_distance)
    sindex = gdf.sindex

    touching_pairs = []
    for idx, poly in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Processing geometries"):
        neighbors_indices = list(sindex.query(poly.geometry, predicate="intersects"))
        for neighbor_idx in neighbors_indices:
            if neighbor_idx != idx:
                neighbor_poly = gdf.iloc[neighbor_idx].geometry
                if check_poly_intersection(poly.geometry, neighbor_poly):
                    touching_pairs.append((idx, neighbor_idx))

    touching_pairs = list(set(frozenset(pair) for pair in touching_pairs))

    # graphe connecte
    G = nx.Graph()

    for idx1, idx2 in touching_pairs:
        G.add_edge(idx1, idx2)

    connected_components = list(nx.connected_components(G))
    connected_components = [list(elt) for elt in connected_components]

    gdf_new = gpd.GeoDataFrame({"filename": pd.Series([])}, geometry=GeoSeries())
    multipolygon_gdf = gpd.GeoDataFrame({"filename": pd.Series([])}, geometry=GeoSeries())

    to_remove = []
    for connected_indexes_poly in connected_components:
        new_poly = Polygon()
        filenames = []
        for idx_poly in connected_indexes_poly:
            new_poly = unary_union([new_poly, gdf.at[idx_poly, "geometry"]])
            filenames.append(gdf.at[idx_poly, "filename"])

        if not isinstance(new_poly, MultiPolygon):
            new_data = gpd.GeoDataFrame({"filename": filenames, "geometry": new_poly})
            gdf_new = pd.concat([gdf_new, new_data], ignore_index=True)
            to_remove += connected_indexes_poly
        else:
            new_data = gpd.GeoDataFrame({"filename": filenames, "geometry": new_poly})
            multipolygon_gdf = pd.concat([multipolygon_gdf, new_data], ignore_index=True)

    gdf = gdf.drop(index=list(set(to_remove)))
    gdf["filename"] = gdf["filename"].apply(lambda x: [x])
    gdf = pd.concat([gdf, gdf_new], ignore_index=True)
    gdf = gdf[~gdf["geometry"].is_empty]
    gdf.reset_index(drop=True, inplace=True)

    gdf["geometry"] = gdf["geometry"].buffer(-buffer_distance)

    return gdf


async def main(dep: str, year: int):
    """
    Perform satellite image inference and save the predictions.

    Args:
        dep (str): The department code.
        year (int): The year of the satellite images.
    """

    # Get info of the model
    model_info = requests.get("https://satellite-images-inference.lab.sspcloud.fr/")

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
    # predictions = merge_adjacent_polygons(predictions)
    predictions_path = f"projet-slums-detection/data-prediction/PLEIADES/{dep}/{year}/{model_info["model_name"]}/{model_info["model_version"]}/predictions"
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
    asyncio.run(main(**args_dict))
