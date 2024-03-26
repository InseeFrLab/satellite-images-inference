import requests
import geopandas as gpd
from tqdm import tqdm
import pandas as pd
import argparse
from s3fs import S3FileSystem
import os
import pyarrow.dataset as ds


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


def main(dep: str, year: int):
    """
    Perform satellite image inference and save the predictions.

    Args:
        dep (str): The department code.
        year (int): The year of the satellite images.
    """

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

    # Request the API to make predictions
    url = "https://satellite-images-inference.lab.sspcloud.fr/predict_image"
    response = [
        requests.get(
            url,
            params={
                "image": image,
                "polygons": True,
            },
        )
        for image in tqdm(images[3:5])
    ]

    predictions = pd.concat(
        [gpd.GeoDataFrame.from_features(r.json()["features"]) for r in response]
    )
    predictions.crs = roi.crs

    predictions_path = (
        f"projet-slums-detection/data-prediction/PLEIADES/{dep}/{year}/predictions.parquet"
    )
    predictions.to_parquet(predictions_path, filesystem=fs)


if __name__ == "__main__":
    args_dict = vars(args)
    main(**args_dict)
