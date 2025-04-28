import geopandas as gpd
import pandas as pd
from geopandas import overlay
from shapely.geometry import box
import requests
import argparse
import time

from make_predictions_from_api import save_geopackage_to_s3
from app.utils import get_file_system


def get_build_evol(
    dep: str, model_name: str, model_version: str, fs
) -> pd.DataFrame:
    global_path = f"projet-slums-detection/data-prediction/PLEIADES/{dep}/"
    paths = fs.ls(global_path, detail=False)

    available_years = []
    pattern = f"/{model_name}/{model_version}/predictions.parquet"

    for path in paths:
        if fs.exists(path+pattern):
            year = path.split("/")[-1]
            available_years.append(int(year))

    data_by_year = {}

    for year in available_years:
        path = f"{global_path}{str(year)}{pattern}"
        df = gpd.read_parquet(path, filesystem=fs)
        df = df[df['label'] == 1][['geometry']].copy()
        # df['geometry'] = df.geometry.apply(lambda geom: box(*geom.bounds))
        df["year"] = year
        data_by_year[year] = df.set_geometry('geometry')

    for year in available_years:
        # Calcul des constructions et destructions
        years_before = [y for y in available_years if y < year]

        if years_before:
            constructions_list = []
            destructions_list = []
            for year_start in years_before:
                year_end = year

                data_start = data_by_year[year_start]
                data_end = data_by_year[year_end]

                data_start = data_start[data_start.is_valid]
                data_end = data_end[data_end.is_valid]

                # Index spatial automatique avec GeoPandas
                # Constructions
                constructions = overlay(data_end, data_start, how='difference')
                constructions['year_start'] = year_start
                constructions['year_end'] = year_end
                constructions_list.append(constructions[['geometry', 'year_start', 'year_end']])

                # Destructions
                destructions = overlay(data_start, data_end, how='difference')
                destructions['year_start'] = year_start
                destructions['year_end'] = year_end
                destructions_list.append(destructions[['geometry', 'year_start', 'year_end']])

            # Assemblage final
            constructions_bati_df = pd.concat(constructions_list, ignore_index=True)
            destructions_bati_df = pd.concat(destructions_list, ignore_index=True)

            return_path = f"""{global_path}{str(year)}/{model_name}/{model_version}/"""

            constructions_bati_df.to_parquet(f"{return_path}constructions.parquet", filesystem=fs)
            save_geopackage_to_s3(
                constructions_bati_df,
                f"{return_path}constructions.gpkg",
                filesystem=fs,
            )

            destructions_bati_df.to_parquet(f"{return_path}destructions.parquet", filesystem=fs)
            save_geopackage_to_s3(
                destructions_bati_df,
                f"{return_path}destructions.gpkg",
                filesystem=fs,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict nuts pipeline")
    parser.add_argument("--dep", type=str, required=True, help="Département (e.g., 'MAYOTTE')")
    args = parser.parse_args()

    dep = args.dep

    start_time = time.time()

    # Get info of the model
    model_info = requests.get("https://satellite-images-inference.lab.sspcloud.fr/").json()

    fs = get_file_system()

    get_build_evol(dep, model_info["model_name"], model_info["model_version"], fs)

    end_time = time.time() - start_time
    print(f"{dep} constructions/destructions in {round(end_time/60)} min")
