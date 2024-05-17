from typing import Union
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
from shapely.geometry import MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union
import libpysal


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
