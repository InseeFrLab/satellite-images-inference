import geopandas as gpd
import unicodedata
import argparse
import pandas as pd

from app.utils.data import get_file_system


def normalize_name(name):
    name_norm = unicodedata.normalize("NFD", name)
    name_norm = "".join(
        char for char in name_norm
        if unicodedata.category(char) != "Mn"
    )
    name_norm = name_norm.replace("'", "")
    name_norm = name_norm.upper()

    return name_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build data-clusters")
    parser.add_argument("--dep_code", type=str, required=True, help="Code du département (e.g., '17')")
    parser.add_argument("--dep_name", type=str, required=True, help="Code du département (e.g., 'CHARENTE-MARITIME')")
    args = parser.parse_args()

    dep_code = args.dep_code
    dep_name = args.dep_name

    fs = get_file_system()
    gdf = gpd.read_file(fs.open("projet-slums-detection/ADE-COG_4-0_GPKG_LAMB93_FXX-ED2026-01-01.gpkg", "rb"))

    dep_codes = dep_code.split("|")
    dep = gdf[gdf["code_insee_du_departement"].isin(dep_codes)]

    epsg = "EPSG:2154"
    dep = dep.to_crs(epsg)

    gdf_clusters = dep[["code_insee_du_departement", "code_insee", "geometry"]]

    gdf_clusters.columns = [
        "dep_code",
        "ident_ilot",
        "geometry"
    ]
    gdf_clusters_wkb = gdf_clusters.copy()
    gdf_clusters_wkb["geometry"] = gpd.GeoSeries.to_wkb(gdf_clusters_wkb["geometry"])

    df_clusters = pd.DataFrame(gdf_clusters_wkb)
    df_clusters["dep"] = dep_name

    df_clusters.to_parquet(
        "data-clusters",
        partition_cols=["dep"],
        index=False
    )
