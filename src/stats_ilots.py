"""
In this file we compute building area statistics in ilots based
on segmentation predictions made by our models.
"""
import geopandas as gpd
from make_predictions_from_api import get_file_system, merge_adjacent_polygons


def compute_ilots_statistics(year: int, dep: str):
    """
    Compute building area statistics in ilots based on segmentation predictions.

    Args:
        year (int): The year of the dataset to make predictions on.
        dep (str): The department of the dataset to make predictions on.
    """
    fs = get_file_system()

    # Load predictions
    predictions_path = f"projet-slums-detection/data-prediction/PLEIADES/{dep}/{year}/predictions"
    with fs.open(f"{predictions_path}.gpkg", "rb") as f:
        predictions = gpd.read_file(f)

    # Remove artifacts
    clean_predictions = merge_adjacent_polygons(predictions)

    # Load ilots data
    with fs.open("projet-slums-detection/ilots/ilots.gpkg", "rb") as f:
        clusters = gpd.read_file(f)

    # Filter ilots
    if dep == "MAYOTTE":
        clusters = clusters.loc[clusters["depcom_2018"].str.startswith("976")]
        # Merge predictions
        clean_predictions = clean_predictions.dissolve()

        # Intersection of ilots with predictions
        clusters = clusters.to_crs("EPSG:4471")
        clusters["geometry"] = clusters["geometry"].intersection(
            clean_predictions.geometry[0]
        )

        # Building area statistics
        clusters["area"] = clusters['geometry'].area / 10**6
        clusters_info = clusters[
            ["ident_ilot", "code", "depcom_2018", "area"]
        ].reset_index(drop=True)

        # Export statistics to csv
        statistics_path = f"projet-slums-detection/prediction_statistics/PLEIADES/ilots_{dep}_{year}.csv"
        with fs.open(statistics_path, "wb") as f:
            clusters_info.to_csv(f, index=False)
    else:
        raise ValueError("Only Mayotte is supported for now.")


if __name__ == "__main__":
    for year in [2017, 2019, 2020, 2022, 2023]:
        compute_ilots_statistics(year, "MAYOTTE")
