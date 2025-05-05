import geopandas as gpd
import pandas as pd
import requests
import argparse
import time
import math

from src.make_predictions_from_api import save_geopackage_to_s3
from app.utils import get_file_system


def filtre_compacite(table: gpd.GeoDataFrame, seuil_compacite: str = 0.08) -> gpd.GeoDataFrame:
    table["compacite"] = (4 * math.pi * table.area) / (table.length**2)
    table_filtree = table[table["compacite"] > seuil_compacite]
    return table_filtree


def get_evolutions(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sym_diff = gpd.overlay(gdf1, gdf2, how="symmetric_difference")
    index = pd.RangeIndex(stop=len(sym_diff))
    sym_diff = gpd.GeoDataFrame(sym_diff, crs=gdf1.crs, index=index)

    polygones_commun = gpd.overlay(gdf1, gdf2, how="intersection")
    resultat = sym_diff[~sym_diff.geometry.isin(polygones_commun.geometry)]

    resultat_1 = gpd.sjoin(resultat, gdf1, how="left", predicate="within")
    suppression = resultat_1[resultat_1.index_right.isna()]
    suppression = suppression.loc[:, resultat.columns]

    resultat_2 = gpd.sjoin(resultat, gdf2, how="left", predicate="within")
    creation = resultat_2[resultat_2.index_right.isna()]
    creation = creation.loc[:, resultat.columns]

    return creation, suppression


def get_predictions(
    dep: str, year: int, model_name: str, model_version: str, fs
) -> gpd.GeoDataFrame:
    path = f"projet-slums-detection/data-prediction/PLEIADES/{dep}/{str(year)}/{model_name}/{model_version}/predictions.parquet"
    data = gpd.read_parquet(path, filesystem=fs)
    data = data[data['label'] == 1][['geometry']].copy()
    data["year"] = year
    data = data.set_geometry('geometry')
    return data


def get_build_evol(
    dep: str, year: int, model_name: str, model_version: str, fs
):
    global_path = f"projet-slums-detection/data-prediction/PLEIADES/{dep}/"
    paths = fs.ls(global_path, detail=False)

    available_years = []
    pattern = f"/{model_name}/{model_version}/predictions.parquet"

    for path in paths:
        if fs.exists(path+pattern):
            year_available = path.split("/")[-1]
            available_years.append(int(year_available))

    data_end = get_predictions(dep, year, model_name, model_version, fs)

    # Calcul des constructions et destructions
    years_before = [y for y in available_years if y < year]

    if years_before:
        constructions_list = []
        destructions_list = []
        for year_start in years_before:
            data_start = get_predictions(dep, year_start, model_name, model_version, fs)

            data_start = data_start[data_start.is_valid]
            data_end = data_end[data_end.is_valid]

            # todo : améliorer la méthode pour filtrer les constructions/destructions

            constructions, destructions = get_evolutions(data_start, data_end)

            constructions = filtre_compacite(constructions)
            destructions = filtre_compacite(destructions)

            constructions["geometry"] = constructions.geometry.buffer(2.5).buffer(-2.5)
            destructions["geometry"] = destructions.geometry.buffer(2.5).buffer(-2.5)

            # Constructions
            constructions['year_start'] = year_start
            constructions['evolution'] = 'construction'
            constructions_list.append(constructions[['geometry', 'year_start', 'evolution']])

            # Destructions
            destructions['year_start'] = year_start
            destructions['evolution'] = 'destruction'
            destructions_list.append(destructions[['geometry', 'year_start', 'evolution']])

        # Assemblage final
        evolutions_bati_df = pd.concat(constructions_list+destructions_list, ignore_index=True)
        evolutions_bati_df['geometry'] = evolutions_bati_df['geometry'].buffer(0)

        return_path = f"""{global_path}{str(year)}/{model_name}/{model_version}/"""

        evolutions_bati_df.to_parquet(f"{return_path}evolutions_bati.parquet", filesystem=fs)
        save_geopackage_to_s3(
            evolutions_bati_df,
            f"{return_path}evolutions_bati.gpkg",
            filesystem=fs,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make constructions/destructions")
    parser.add_argument("--dep", type=str, required=True, help="Département (e.g., 'MAYOTTE')")
    parser.add_argument("--year", type=int, required=True, help="Année (e.g., 2017)")
    args = parser.parse_args()

    dep = args.dep
    year = args.year

    start_time = time.time()

    # Get info of the model
    model_info = requests.get("https://satellite-images-inference.lab.sspcloud.fr/").json()

    fs = get_file_system()

    get_build_evol(dep, year, model_info["model_name"], model_info["model_version"], fs)

    end_time = time.time() - start_time
    print(f"{dep} constructions/destructions in {round(end_time/60)} min")
