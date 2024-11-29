"""
Script to analyze building area statistics in Mayotte.
"""
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.dataset as ds
import rasterio
from make_predictions_from_api import get_file_system, merge_adjacent_polygons
from matplotlib import pyplot as plt
from osgeo import gdal
from s3fs import S3FileSystem


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


def main():
    """
    Analyze building area statistics in Mayotte.
    """
    gdal.SetConfigOption("GDAL_NUM_THREADS", "1")

    fs = get_file_system()
    # Load statistics from 2020
    with fs.open(
        "projet-slums-detection/prediction_statistics/PLEIADES/ilots_MAYOTTE_2020.csv", "rb"
    ) as f:
        statistics_2020 = pd.read_csv(f)
    # Load statistics from 2023
    with fs.open(
        "projet-slums-detection/prediction_statistics/PLEIADES/ilots_MAYOTTE_2023.csv", "rb"
    ) as f:
        statistics_2023 = pd.read_csv(f)

    # Merge statistics
    df = pd.merge(
        statistics_2020,
        statistics_2023,
        on=["ident_ilot", "code", "depcom_2018"],
        suffixes=("_2020", "_2023"),
    )

    # Compute difference
    df["area_diff"] = df["area_2023"] - df["area_2020"]

    # Get 5 ilots with the largest area difference
    target_clusters = df.nlargest(5, "area_diff").ident_ilot.astype(str).tolist()
    # We would like to plot the tiles corresponding to these
    # ilots along with predictions for 2020 and 2023.
    with fs.open("projet-slums-detection/ilots/ilots.gpkg", "rb") as f:
        clusters = gpd.read_file(f)

    for year in [2020, 2023]:
        predictions_path = (
            f"projet-slums-detection/data-prediction/PLEIADES/MAYOTTE/{year}/predictions"
        )
        with fs.open(f"{predictions_path}.gpkg", "rb") as f:
            predictions = gpd.read_file(f)
        # Remove artifacts
        clean_predictions = merge_adjacent_polygons(predictions)
        for target_cluster in target_clusters:
            # Get the filename to polygons mapping
            filename_table = get_filename_to_polygons("MAYOTTE", year, fs)

            # Get the selected cluster
            selected_cluster = (
                clusters.loc[clusters["ident_ilot"] == target_cluster]
                .to_crs("EPSG:4471")
                .geometry.iloc[0]
            )

            # Get predictions
            selected_predictions = clean_predictions.loc[
                clean_predictions.geometry.intersects(selected_cluster)
            ]
            selected_predictions["geometry"] = selected_predictions["geometry"].intersection(
                selected_cluster
            )

            # Get the filenames of the images that intersect with the selected cluster
            image_paths = filename_table.loc[
                filename_table.geometry.intersects(selected_cluster),
                "filename",
            ].tolist()
            print(
                f"{len(image_paths)} images intersect with the selected cluster {target_cluster}."
            )

            # Plot mosaic
            fig, ax = plt.subplots()
            with tempfile.TemporaryDirectory() as tmpdir:
                for image_path in image_paths:
                    file_name = Path(image_path).name
                    fs.get(image_path, f"{tmpdir}/{file_name}")
                    raster = rasterio.open(f"{tmpdir}/{file_name}")
                    rasterio.plot.show(raster, ax=ax)
            # Plot the selected cluster
            gpd.GeoSeries([selected_cluster]).plot(
                ax=ax, facecolor="none", edgecolor="red", linewidth=0.1
            )
            # Plot buildings
            selected_predictions["geometry"].plot(
                ax=ax, facecolor="none", edgecolor="cyan", linewidth=0.1
            )
            fig.savefig(f"preds_{target_cluster}_{year}.png", dpi=500)


if __name__ == "__main__":
    main()
