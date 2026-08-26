import geopandas as gpd
import unicodedata
import argparse

from app.utils.utils import get_file_system


dep_code_to_crs = {
        "971": "EPSG:4559",
        "972": "EPSG:4559",
        "973": "EPSG:2972",
        "974": "EPSG:2975",
        "976": "EPSG:4471",
        "977": "EPSG:4559",
        "978": "EPSG:4559",
    }


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
    parser = argparse.ArgumentParser(description="Build data-roi")
    parser.add_argument("--dep_code", type=str, required=True, help="Code du département (e.g., '75')")
    args = parser.parse_args()

    dep_code = args.dep_code

    fs = get_file_system()
    gdf = gpd.read_file(fs.open("projet-slums-detection/departements-50m.geojson", "rb"))

    dep_codes = dep_code.split("|")
    dep = gdf[gdf["code"].isin(dep_codes)]

    all_dep_names = dep.nom.values
    all_dep_names = [normalize_name(dep_name) for dep_name in all_dep_names]
    sets_de_mots = [set(s.split('-')) for s in all_dep_names]
    mots_communs = list(set.intersection(*sets_de_mots))
    dep_name = ("-").join(mots_communs)
    dep_name_norm = normalize_name(dep_name)

    epsg = "EPSG:2154"

    if dep_code in dep_code_to_crs.keys():
        epsg = dep_code_to_crs[dep_code]

    dep = dep.to_crs(epsg)

    dep_merge = dep.dissolve(
        by="region",
        as_index=False
    )

    gdf_roi = gpd.GeoDataFrame(
        {
            "ID": ["DEPARTEMENT_"+dep.code],
            "NOM_M": [dep_name_norm],
            "NOM": [dep_name],
            "INSEE_REG": [dep.region],
            "geometry": [dep.geometry.values[0]],
        },
        crs=epsg,
    )

    gdf_roi.to_file(
        dep_name_norm+".geojson",
        driver="GeoJSON"
    )
